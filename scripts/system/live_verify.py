"""
scripts.system.live_verify — Live system verification harness.

Preferred (tokens not on argv / shell history):

  export LIVE_VERIFY_TOKEN_A=… LIVE_VERIFY_TOKEN_B=…
  python -m scripts.system.live_verify --base-url https://api.example.com

Optional: --song-a / --song-b for an end-to-end remix timing sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class VerifyCheck:
    """Single verification check result."""

    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{status} {self.name}: {self.detail}"


def _token_fingerprint(token: str) -> str:
    """Short non-reversible id for logs (avoids identical eyJhbGci prefixes)."""
    if not token:
        return "<none>"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def _scrub_detail(text: str, limit: int = 200) -> str:
    scrubbed = re.sub(r"(?i)bearer\s+\S+", "Bearer <redacted>", text or "")
    return scrubbed[:limit]


class _NoCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that would forward Authorization to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        old_host = urllib.parse.urlsplit(req.full_url).netloc
        new_host = urllib.parse.urlsplit(newurl).netloc
        if new_host != old_host:
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(
    urllib.request.HTTPHandler(),
    urllib.request.HTTPSHandler(),
    _NoCrossHostRedirect(),
)


def _make_request(
    url: str,
    method: str = "GET",
    token: Optional[str] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Make HTTP request with optional auth. Returns (status_code, json_body)."""
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=10) as response:
            body_bytes = response.read(1_000_000)
            if not body_bytes:
                return response.status, {}
            try:
                parsed = json.loads(body_bytes)
            except json.JSONDecodeError:
                return response.status, {
                    "detail": _scrub_detail(body_bytes.decode("utf-8", errors="replace"))
                }
            return response.status, parsed if isinstance(parsed, dict) else {"data": parsed}
    except urllib.error.HTTPError as e:
        body_bytes = e.read(1_000_000)
        try:
            body_json = json.loads(body_bytes) if body_bytes else {}
            if not isinstance(body_json, dict):
                body_json = {"detail": str(body_json)}
        except json.JSONDecodeError:
            body_json = {
                "detail": _scrub_detail(body_bytes.decode("utf-8", errors="replace"))
            }
        return e.code, body_json
    except Exception as e:  # noqa: BLE001
        return 0, {"detail": _scrub_detail(str(e))}


def _validate_base_url(base_url: str) -> Optional[str]:
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in ("http", "https"):
        return "base-url scheme must be http or https"
    host = (parts.hostname or "").lower()
    loopback = host in ("127.0.0.1", "localhost", "::1")
    if parts.scheme != "https" and not loopback:
        return "base-url must be https for non-loopback hosts"
    return None


def check_health_ready(base_url: str) -> VerifyCheck:
    status, body = _make_request(f"{base_url.rstrip('/')}/health/ready")
    passed = status == 200
    detail = f"{status}" if passed else f"{status} {_scrub_detail(str(body.get('detail', body)))}"
    return VerifyCheck("health_ready", passed, detail)


_FEATURE_KEYS = (
    "auth",
    "stem_split",
    "generative",
    "license_enforce",
    "spectral_learning",
    "operator",
)
_SSE_WINDOW_SEC = 10.0


def check_features(base_url: str) -> Tuple[VerifyCheck, Optional[bool]]:
    """GET /features; returns the check and the auth flag (None when unknown)."""
    status, body = _make_request(f"{base_url.rstrip('/')}/features")
    if status != 200:
        detail = f"{status} {_scrub_detail(str(body.get('detail', '')))}"
        return VerifyCheck("features", False, detail), None
    missing = [k for k in _FEATURE_KEYS if k not in body]
    if missing:
        return VerifyCheck("features", False, f"missing keys: {', '.join(missing)}"), None
    auth_on = body["auth"] is not False
    return VerifyCheck("features", True, f"200 auth={'on' if auth_on else 'off'}"), auth_on


def _job_list(body: Any) -> Optional[List[Any]]:
    """GET /jobs answers a JSON array, which _make_request wraps as {"data": [...]}."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for key in ("data", "jobs"):
            if isinstance(body.get(key), list):
                return body[key]
    return None


def check_auth_works(base_url: str, token: str, label: str) -> VerifyCheck:
    status, body = _make_request(f"{base_url.rstrip('/')}/jobs?limit=5", token=token)
    passed = status == 200
    if passed:
        jobs = _job_list(body)
        n = len(jobs) if jobs is not None else 0
        detail = f"{label} auth ok ({n} jobs)"
    else:
        detail = f"{label} {status} {_scrub_detail(str(body.get('detail', '')))}"
    return VerifyCheck(f"auth_{label.lower().replace(' ', '_')}", passed, detail)


def _crate_ids(body: Dict[str, Any]) -> Optional[List[str]]:
    """Extract crate_id list; None if response shape is wrong."""
    if not isinstance(body, dict) or "crates" not in body:
        return None
    crates = body.get("crates")
    if not isinstance(crates, list):
        return None
    return [c.get("crate_id") for c in crates if isinstance(c, dict)]


def check_cross_tenant_isolation(
    base_url: str, token_a: str, token_b: str
) -> VerifyCheck:
    """Create crate as A; require A sees it and B does not."""
    base = base_url.rstrip("/")
    crate_name = f"verify_test_{uuid.uuid4().hex[:12]}"
    status, body = _make_request(
        f"{base}/crates",
        method="POST",
        token=token_a,
        body={"name": crate_name},
    )
    if status != 201:
        return VerifyCheck(
            "cross_tenant_isolation",
            False,
            f"Failed to create crate: {status} {_scrub_detail(str(body.get('detail', '')))}",
        )

    crate_id = body.get("crate_id")
    if not crate_id or not isinstance(crate_id, str):
        return VerifyCheck("cross_tenant_isolation", False, "No crate_id in create response")

    qid = urllib.parse.quote(crate_id, safe="")
    try:
        status_a, body_a = _make_request(f"{base}/crates", token=token_a)
        ids_a = _crate_ids(body_a)
        if status_a != 200 or ids_a is None:
            return VerifyCheck(
                "cross_tenant_isolation",
                False,
                f"Unexpected /crates shape from A: {status_a}",
            )
        if crate_id not in ids_a:
            return VerifyCheck(
                "cross_tenant_isolation",
                False,
                "Positive control failed: A cannot see its own crate",
            )

        status_b, body_b = _make_request(f"{base}/crates", token=token_b)
        ids_b = _crate_ids(body_b)
        if status_b != 200 or ids_b is None:
            return VerifyCheck(
                "cross_tenant_isolation",
                False,
                f"Unexpected /crates shape from B: {status_b}",
            )
        if crate_id in ids_b:
            return VerifyCheck(
                "cross_tenant_isolation",
                False,
                f"Token B sees crate {crate_id[:8]} (LEAK)",
            )

        status_songs, body_songs = _make_request(
            f"{base}/crates/{qid}/songs", token=token_b
        )
        songs = body_songs.get("songs") if isinstance(body_songs, dict) else None
        if status_songs == 200 and isinstance(songs, list) and len(songs) > 0:
            return VerifyCheck(
                "cross_tenant_isolation",
                False,
                "Token B listed songs on A's crate (LEAK)",
            )

        return VerifyCheck(
            "cross_tenant_isolation",
            True,
            f"Token B correctly excluded crate {crate_id[:8]}",
        )
    finally:
        _make_request(f"{base}/crates/{qid}", method="DELETE", token=token_a)


def _create_job(base: str, token: str) -> Tuple[Optional[str], str]:
    """POST /analyze on the first GET /library song. Returns (job_id, error)."""
    status, body = _make_request(f"{base}/library?per_page=1", token=token)
    songs = body.get("songs") if status == 200 and isinstance(body, dict) else None
    name = next(
        (s.get("name") for s in songs or [] if isinstance(s, dict) and s.get("name")),
        None,
    )
    if not name:
        return None, f"No library song to create a job: {status}"
    status, body = _make_request(
        f"{base}/analyze", method="POST", token=token, body={"song": name}
    )
    job_id = body.get("job_id") if isinstance(body, dict) else None
    if status not in (200, 202) or not job_id:
        return None, (
            f"Failed to create job: {status} {_scrub_detail(str(body.get('detail', '')))}"
        )
    return str(job_id), ""


def check_job_by_id_isolation(
    base_url: str, token_a: str, token_b: str
) -> VerifyCheck:
    """A owns a job; require A reads it by id and B does not."""
    name = "job_by_id_isolation"
    base = base_url.rstrip("/")
    status, body = _make_request(f"{base}/jobs?limit=5", token=token_a)
    jobs = _job_list(body) if status == 200 else None
    job_id = next(
        (str(j["job_id"]) for j in jobs or [] if isinstance(j, dict) and j.get("job_id")),
        None,
    )
    if not job_id:
        job_id, err = _create_job(base, token_a)
        if not job_id:
            return VerifyCheck(name, False, err)

    qid = urllib.parse.quote(job_id, safe="")
    status_a, _ = _make_request(f"{base}/jobs/{qid}", token=token_a)
    if status_a != 200:
        return VerifyCheck(
            name, False, f"Positive control failed: A cannot read its own job ({status_a})"
        )
    status_b, _ = _make_request(f"{base}/jobs/{qid}", token=token_b)
    if status_b == 200:
        return VerifyCheck(name, False, f"Token B read job {job_id[:8]} by id (LEAK)")
    if status_b not in (403, 404):
        return VerifyCheck(name, False, f"Unexpected status for Token B: {status_b}")
    return VerifyCheck(name, True, f"Token B got {status_b} for job {job_id[:8]}")


def _read_sse(
    base: str,
    token: str,
    window: float,
    connected: threading.Event,
    out: Dict[str, Any],
) -> None:
    """Collect JSON data frames from /events/stream until EOF or the window ends."""
    frames: List[Dict[str, Any]] = []
    out["frames"] = frames
    req = urllib.request.Request(
        f"{base}/events/stream",
        headers={"Accept": "text/event-stream", "Authorization": f"Bearer {token}"},
    )
    try:
        resp = _OPENER.open(req, timeout=window)
    except OSError as e:
        out["error"] = _scrub_detail(str(e))
        connected.set()
        return
    deadline = time.monotonic() + window
    try:
        with resp:
            while time.monotonic() < deadline:
                try:
                    line = resp.readline()
                except TimeoutError:
                    break
                connected.set()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text.startswith("data:"):
                    continue
                try:
                    frame = json.loads(text[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(frame, dict):
                    frames.append(frame)
    except OSError as e:
        out["error"] = _scrub_detail(str(e))
    finally:
        connected.set()


def _sse_job_ids(frames: List[Dict[str, Any]]) -> set:
    return {
        str(f["data"].get("job_id"))
        for f in frames
        if str(f.get("type", "")).startswith("job_") and isinstance(f.get("data"), dict)
    }


def check_sse_isolation(
    base_url: str, token_a: str, token_b: str, window: float = _SSE_WINDOW_SEC
) -> VerifyCheck:
    """A and B hold /events/stream while A creates a job; B must not see its frames."""
    name = "sse_isolation"
    base = base_url.rstrip("/")
    results: Dict[str, Dict[str, Any]] = {"A": {}, "B": {}}
    events = {"A": threading.Event(), "B": threading.Event()}
    threads = [
        threading.Thread(
            target=_read_sse,
            args=(base, token, window, events[label], results[label]),
            daemon=True,
        )
        for label, token in (("A", token_a), ("B", token_b))
    ]
    for t in threads:
        t.start()
    for ev in events.values():
        ev.wait(window)
    for label in ("A", "B"):
        if "error" in results[label]:
            return VerifyCheck(name, False, f"Token {label} stream: {results[label]['error']}")

    job_id, err = _create_job(base, token_a)
    for t in threads:
        t.join(window + 5)
    if not job_id:
        return VerifyCheck(name, False, err)
    for label in ("A", "B"):
        if "error" in results[label]:
            return VerifyCheck(name, False, f"Token {label} stream: {results[label]['error']}")

    if job_id in _sse_job_ids(results["B"].get("frames", [])):
        return VerifyCheck(name, False, f"Token B received job {job_id[:8]} events (LEAK)")
    if job_id not in _sse_job_ids(results["A"].get("frames", [])):
        return VerifyCheck(
            name, False, "Positive control failed: A's stream carried no event for its job"
        )
    return VerifyCheck(name, True, f"Token B saw no events for job {job_id[:8]}")


def check_render_job(
    base_url: str,
    token: str,
    song_a: str,
    song_b: str,
    timeout: int,
) -> VerifyCheck:
    base = base_url.rstrip("/")
    status, body = _make_request(
        f"{base}/dj-remix",
        method="POST",
        token=token,
        body={
            "song_a": song_a,
            "song_b": song_b,
            "transition_bars": 16,
            "preset": "auto",
            "bridge_beat_mode": "none",
            "bridge_beat_genre": "auto",
            "bridge_beat_intensity": 0.38,
            "bridge_beat_path": None,
            "transition_effect": "none",
        },
    )
    if status != 202:
        return VerifyCheck(
            "render_job",
            False,
            f"Failed to submit job: {status} {_scrub_detail(str(body.get('detail', '')))}",
        )

    job_id = body.get("job_id")
    if not job_id:
        return VerifyCheck("render_job", False, "No job_id in response")

    qid = urllib.parse.quote(str(job_id), safe="")
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            return VerifyCheck(
                "render_job", False, f"Timeout after {int(elapsed)}s (limit {timeout}s)"
            )

        status_poll, job_body = _make_request(f"{base}/jobs/{qid}", token=token)
        if status_poll != 200:
            return VerifyCheck(
                "render_job",
                False,
                f"Poll failed: {status_poll} {_scrub_detail(str(job_body.get('detail', '')))}",
            )

        raw = job_body.get("status", "")
        job_status = str(raw).split(".")[-1].upper()
        if job_status in ("COMPLETED", "DONE"):
            result = job_body.get("result") or {}
            has_credits = isinstance(result, dict) and (
                "credits" in result
                or any("CREDITS" in str(v) for v in result.values())
            )
            detail = f"Job completed in {int(elapsed)}s"
            detail += ", credits present" if has_credits else ", no credits field"
            return VerifyCheck("render_job", True, detail)
        if job_status == "FAILED":
            error = _scrub_detail(str(job_body.get("error", "unknown error")))
            return VerifyCheck("render_job", False, f"Job failed: {error}")
        time.sleep(2.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live system verification for RemixMate API"
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="API base URL"
    )
    parser.add_argument(
        "--token-a",
        default=os.getenv("LIVE_VERIFY_TOKEN_A"),
        help="Prefer LIVE_VERIFY_TOKEN_A env (avoid shell history)",
    )
    parser.add_argument(
        "--token-b",
        default=os.getenv("LIVE_VERIFY_TOKEN_B"),
        help="Prefer LIVE_VERIFY_TOKEN_B env (avoid shell history)",
    )
    parser.add_argument("--token-a-file", help="Read token A from a file")
    parser.add_argument("--token-b-file", help="Read token B from a file")
    parser.add_argument("--song-a", help="Optional catalog song for remix test")
    parser.add_argument("--song-b", help="Optional catalog song for remix test")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Job poll timeout in seconds"
    )
    parser.add_argument(
        "--skip-render", action="store_true", help="Skip render job check"
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="Print JSON results"
    )

    args = parser.parse_args()

    err = _validate_base_url(args.base_url)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    token_a = (
        Path(args.token_a_file).read_text(encoding="utf-8").strip()
        if args.token_a_file
        else args.token_a
    )
    token_b = (
        Path(args.token_b_file).read_text(encoding="utf-8").strip()
        if args.token_b_file
        else args.token_b
    )
    if not token_a or not token_b:
        print(
            "Error: set LIVE_VERIFY_TOKEN_A/B or --token-a-file/--token-b-file",
            file=sys.stderr,
        )
        return 1
    if token_a == token_b:
        print("Error: token A and B must be different users", file=sys.stderr)
        return 1

    if not args.as_json:
        print(f"RemixMate Live Verify — {args.base_url}")
        print(f"Token A fp: {_token_fingerprint(token_a)}")
        print(f"Token B fp: {_token_fingerprint(token_b)}")
        print()

    features_check, auth_on = check_features(args.base_url)
    checks: List[VerifyCheck] = [
        check_health_ready(args.base_url),
        features_check,
        check_auth_works(args.base_url, token_a, "Token A"),
        check_auth_works(args.base_url, token_b, "Token B"),
    ]
    if auth_on is False:
        # Auth off resolves every token to one local user, so isolation would read as a leak.
        if not args.as_json:
            print("SKIP cross_tenant_isolation (auth disabled)")
            print("SKIP job_by_id_isolation (auth disabled)")
            print("SKIP sse_isolation (auth disabled)\n")
    else:
        checks.append(check_cross_tenant_isolation(args.base_url, token_a, token_b))
        checks.append(check_job_by_id_isolation(args.base_url, token_a, token_b))
        checks.append(check_sse_isolation(args.base_url, token_a, token_b))

    if not args.skip_render and args.song_a and args.song_b:
        checks.append(
            check_render_job(
                args.base_url, token_a, args.song_a, args.song_b, args.timeout
            )
        )
    elif not args.skip_render and not args.as_json and (not args.song_a or not args.song_b):
        print("SKIP render_job: no songs provided\n")

    if args.as_json:
        print(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "detail": c.detail}
                        for c in checks
                    ],
                    "all_passed": all(c.passed for c in checks),
                },
                indent=2,
            )
        )
    else:
        for check in checks:
            print(check)
        print()
        if all(c.passed for c in checks):
            print("All checks passed")
        else:
            failed = [c for c in checks if not c.passed]
            print(f"{len(failed)} check(s) failed")

    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
