"""tests/test_live_verify.py — mocked HTTP checks for the live verify harness.

B8.4 contract (plans/audit-the-project-and-sorted-flask.md, B8.4; SSE semantics
from B6.3, line 791; defects from line 918). The builder implements against it:

* Auth state is read from ``GET /features`` (``auth`` key, B5.1; the only place
  the API exposes it). When it is false, ``main()`` prints exactly
  ``SKIP cross_tenant_isolation (auth disabled)``, runs no tenant isolation
  check (auth off resolves every token to ``local``, so any isolation check
  would report a leak) and exits 0.
* New checks, by their ``name`` in the check list:
    ``features``            GET /features answers 200 with the six B5.1 keys;
                            a 404 is a failed row, not an abort (JSON still printed).
    ``job_by_id_isolation`` A owns a job (from GET /jobs as A, or one A creates);
                            GET /jobs/{id} as B must not answer 200 (the API
                            answers 404). A 200 for B is a LEAK.
    ``sse_isolation``       B is connected to GET /events/stream (Bearer B) while
                            A creates a job (POST /analyze on a GET /library song,
                            or POST /dj-remix); B's stream carries no ``job_*``
                            frame with A's job id. A's own stream is the positive
                            control. A leaked frame is a LEAK.
* Job count display: GET /jobs returns a JSON array, which ``_make_request``
  wraps as ``{"data": [...]}``; the count is ``len(body["data"])``, not
  ``len(body)`` (which is always 1).

The B8.4 tests fake the HTTP layer at ``lv._OPENER.open`` and
``urllib.request.urlopen`` so the real ``_make_request`` runs and any helper
that goes through the module opener is intercepted. Nothing reaches the network.
The fake SSE stream yields the ack frame, waits briefly for frames, sends one
``: ping`` comment and then EOF; a check must treat EOF as the end of its
window, not as an error.
"""

from __future__ import annotations

import io
import json
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from unittest import mock

from scripts.system import live_verify as lv

TOKEN_A = "token-user-a-aaaaaaaa"
TOKEN_B = "token-user-b-bbbbbbbb"


class LiveVerifyTests(unittest.TestCase):
    def test_health_ready_pass(self):
        with mock.patch.object(lv, "_make_request", return_value=(200, {"status": "ready"})):
            check = lv.check_health_ready("http://127.0.0.1:8000")
        self.assertTrue(check.passed)

    def test_cross_tenant_leak_fails(self):
        crate_id = "crate_A"

        def fake(url, method="GET", token=None, body=None):
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": crate_id, "name": "t"}
            if method == "GET" and url.endswith("/crates"):
                if token == TOKEN_A:
                    return 200, {"crates": [{"crate_id": crate_id}]}
                return 200, {"crates": [{"crate_id": crate_id}]}  # leak
            if method == "DELETE":
                return 200, {"deleted": True}
            return 404, {}

        with mock.patch.object(lv, "_make_request", side_effect=fake):
            check = lv.check_cross_tenant_isolation(
                "http://127.0.0.1:8000", TOKEN_A, TOKEN_B
            )
        self.assertFalse(check.passed)
        self.assertIn("LEAK", check.detail)

    def test_cross_tenant_isolated_pass(self):
        crate_id = "crate_A"

        def fake(url, method="GET", token=None, body=None):
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": crate_id, "name": "t"}
            if method == "GET" and url.endswith("/crates"):
                if token == TOKEN_A:
                    return 200, {"crates": [{"crate_id": crate_id}]}
                return 200, {"crates": []}
            if "/songs" in url:
                return 200, {"songs": []}
            if method == "DELETE":
                return 200, {"deleted": True}
            return 404, {}

        with mock.patch.object(lv, "_make_request", side_effect=fake):
            check = lv.check_cross_tenant_isolation(
                "http://127.0.0.1:8000", TOKEN_A, TOKEN_B
            )
        self.assertTrue(check.passed)

    def test_vacuous_shape_does_not_pass(self):
        """B response missing crates key must fail (not pass vacuously)."""
        crate_id = "crate_A"

        def fake(url, method="GET", token=None, body=None):
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": crate_id}
            if method == "GET" and url.endswith("/crates"):
                if token == TOKEN_A:
                    return 200, {"crates": [{"crate_id": crate_id}]}
                return 200, {"items": [{"crate_id": crate_id}]}  # wrong shape + leak
            if method == "DELETE":
                return 200, {}
            return 404, {}

        with mock.patch.object(lv, "_make_request", side_effect=fake):
            check = lv.check_cross_tenant_isolation(
                "http://127.0.0.1:8000", TOKEN_A, TOKEN_B
            )
        self.assertFalse(check.passed)
        self.assertIn("shape", check.detail.lower())

    def test_positive_control_required(self):
        crate_id = "crate_A"

        def fake(url, method="GET", token=None, body=None):
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": crate_id}
            if method == "GET" and url.endswith("/crates"):
                return 200, {"crates": []}  # A can't see own crate
            if method == "DELETE":
                return 200, {}
            return 404, {}

        with mock.patch.object(lv, "_make_request", side_effect=fake):
            check = lv.check_cross_tenant_isolation(
                "http://127.0.0.1:8000", TOKEN_A, TOKEN_B
            )
        self.assertFalse(check.passed)
        self.assertIn("Positive control", check.detail)

    def test_validate_base_url_requires_https_remote(self):
        self.assertIsNotNone(lv._validate_base_url("http://api.example.com"))
        self.assertIsNone(lv._validate_base_url("https://api.example.com"))
        self.assertIsNone(lv._validate_base_url("http://127.0.0.1:8000"))

    def test_no_raw_jwt_in_stdout(self):
        jwtish = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        fake = FakeApi(token_a=jwtish, token_b=jwtish + "b")
        code, out, err = _run_main(fake)
        self.assertEqual(code, 0, msg=f"stdout={out!r} stderr={err!r}")
        self.assertNotIn(jwtish, out)
        self.assertNotIn(jwtish + "b", out)

    def test_json_mode(self):
        code, out, err = _run_main(FakeApi(), "--json")
        data = json.loads(out)
        self.assertEqual(code, 0, msg=f"stdout={out!r} stderr={err!r}")
        self.assertTrue(data["all_passed"])


# ---------------------------------------------------------------------------
# B8.4: a fake API at the opener level (real _make_request runs on top of it)
# ---------------------------------------------------------------------------

USER_A, USER_B, LOCAL = "user_a", "user_b", "local"
ALL_USERS = (USER_A, USER_B, LOCAL)
FEATURE_KEYS = {"auth", "stem_split", "generative", "license_enforce", "spectral_learning", "operator"}
LIBRARY_SONGS = ["Song One", "Song Two"]
BASE = "http://127.0.0.1:8000"
SKIP_LINE = "SKIP cross_tenant_isolation (auth disabled)"
NEW_CHECKS = {"features", "job_by_id_isolation", "sse_isolation"}


def _http_error(url: str, code: int, body: dict) -> urllib.error.HTTPError:
    hdrs = Message()
    hdrs["Content-Type"] = "application/json"
    return urllib.error.HTTPError(url, code, "error", hdrs, io.BytesIO(json.dumps(body).encode("utf-8")))


class _FakeResponse:
    def __init__(self, status: int, payload, content_type: str = "application/json"):
        self.status = status
        self.code = status
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def read(self, amt=-1):
        return self._buf.read(amt)

    def readline(self, limit=-1):
        return self._buf.readline(limit)

    def __iter__(self):
        return iter(self._buf.readline, b"")

    def getcode(self):
        return self.status

    def info(self):
        return self.headers

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeStream:
    """text/event-stream for one user: ack, then frames the fake emits to that
    user (waiting briefly for late ones), one ``: ping`` comment, then EOF."""

    def __init__(self, api: "FakeApi", user: str):
        self.api = api
        self.user = user
        self.status = 200
        self.code = 200
        self.headers = Message()
        self.headers["Content-Type"] = "text/event-stream"
        self._pending = b""
        self._sent_ack = False
        self._cursor = 0
        self._idle = 0
        self._closed = False

    def _next(self) -> bytes:
        if self._closed:
            return b""
        if not self._sent_ack:
            self._sent_ack = True
            ack = {"type": "system_status", "data": {"status": "ok", "message": "SSE stream connected"}, "ts": "2026-09-28T00:00:00.000Z"}
            return b"data: " + json.dumps(ack).encode("utf-8") + b"\n\n"
        with self.api.cond:
            frames = self.api.frames.setdefault(self.user, [])
            if self._cursor >= len(frames):
                self.api.cond.wait(timeout=0.3)
            if self._cursor < len(frames):
                frame = frames[self._cursor]
                self._cursor += 1
                return frame
        self._idle += 1
        if self._idle == 1:
            return b": ping\n\n"
        self._closed = True
        return b""

    def readline(self, limit=-1):
        while b"\n" not in self._pending:
            chunk = self._next()
            if not chunk:
                line, self._pending = self._pending, b""
                return line
            self._pending += chunk
        i = self._pending.index(b"\n") + 1
        line, self._pending = self._pending[:i], self._pending[i:]
        return line

    def read(self, amt=-1):
        if amt is None or amt < 0:
            out, self._pending = self._pending, b""
            while True:
                chunk = self._next()
                if not chunk:
                    return out
                out += chunk
        while len(self._pending) < amt:
            chunk = self._next()
            if not chunk:
                break
            self._pending += chunk
        out, self._pending = self._pending[:amt], self._pending[amt:]
        return out

    def __iter__(self):
        return iter(self.readline, b"")

    def getcode(self):
        return 200

    def info(self):
        return self.headers

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class FakeApi:
    """Owner scoped fake of the routes live_verify touches.

    auth_on=False models a local box: every token resolves to ``local``, so
    any isolation check that still runs sees a leak (plan line 918).
    leak_jobs: GET /jobs/{id} answers 200 to a non owner (the defect).
    leak_sse: a job frame reaches every connected stream, not only the owner's.
    """

    def __init__(self, *, auth_on=True, leak_jobs=False, leak_sse=False, features_present=True, jobs_a=1, jobs_b=0,
                 token_a=TOKEN_A, token_b=TOKEN_B):
        self.auth_on = auth_on
        self.token_a = token_a
        self.token_b = token_b
        self.leak_jobs = leak_jobs
        self.leak_sse = leak_sse
        self.features_present = features_present
        self.cond = threading.Condition()
        self.frames: dict[str, list[bytes]] = {}
        self.calls: list[tuple[str, str, str | None]] = []
        self.crates: list[dict] = []
        self.jobs: list[dict] = []
        self.created: list[dict] = []
        owner_a, owner_b = (USER_A, USER_B) if auth_on else (LOCAL, LOCAL)
        for i in range(jobs_a):
            self.jobs.append(self._job(f"job_a{i + 1}", owner_a))
        for i in range(jobs_b):
            self.jobs.append(self._job(f"job_b{i + 1}", owner_b))

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _job(job_id: str, owner: str, job_type: str = "analyze") -> dict:
        return {"job_id": job_id, "owner": owner, "status": "COMPLETED", "job_type": job_type}

    @staticmethod
    def _public(job: dict) -> dict:
        return {
            "job_id": job["job_id"], "status": job["status"], "job_type": job["job_type"],
            "created_at": 1.0, "started_at": None, "finished_at": None, "progress": 1.0,
            "message": "", "result": None, "error": None, "eta_sec": None,
        }

    def _user(self, token):
        if not self.auth_on:
            return LOCAL
        return {self.token_a: USER_A, self.token_b: USER_B}.get(token)

    def _create_job(self, owner: str, job_type: str) -> dict:
        job = self._job(f"job_new{len(self.created) + 1}", owner, job_type)
        job["status"] = "PENDING"
        self.jobs.append(job)
        self.created.append(job)
        payload = self._public(job)
        payload["type"] = job_type
        payload["meta"] = {"user_id": owner}
        frame = b"data: " + json.dumps({"type": "job_created", "data": payload, "ts": "2026-09-28T00:00:01.000Z"}).encode("utf-8") + b"\n\n"
        with self.cond:
            targets = ALL_USERS if self.leak_sse else (owner,)
            for u in targets:
                self.frames.setdefault(u, []).append(frame)
            self.cond.notify_all()
        return job

    def jobs_created_by(self, owner: str) -> list[dict]:
        return [j for j in self.created if j["owner"] == owner]

    def foreign_job_reads(self, token: str) -> list[str]:
        """Job ids requested by id with `token` that belong to another user."""
        me = self._user(token)
        owners = {j["job_id"]: j["owner"] for j in self.jobs}
        out = []
        for method, path, tok in self.calls:
            if method == "GET" and tok == token and path.startswith("/jobs/"):
                jid = urllib.parse.unquote(path[len("/jobs/"):])
                if jid in owners and owners[jid] != me:
                    out.append(jid)
        return out

    # -- dispatcher ---------------------------------------------------------
    def open(self, req, data=None, timeout=None, **_kw):
        if isinstance(req, str):
            req = urllib.request.Request(req, data=data)
        url = req.full_url
        parts = urllib.parse.urlsplit(url)
        path = parts.path.rstrip("/") or "/"
        method = req.get_method()
        auth_header = req.get_header("Authorization") or ""
        token = auth_header[len("Bearer "):] if auth_header.startswith("Bearer ") else None
        self.calls.append((method, path, token))
        raw = req.data if req.data is not None else data
        body = json.loads(raw) if raw else None

        if path == "/health/ready":
            return _FakeResponse(200, {"status": "ready"})
        if path == "/features":
            if not self.features_present:
                raise _http_error(url, 404, {"detail": "Not Found"})
            return _FakeResponse(200, {
                "auth": self.auth_on, "stem_split": False, "generative": False,
                "license_enforce": True, "spectral_learning": True, "operator": not self.auth_on,
            })

        user = self._user(token)
        if user is None:
            raise _http_error(url, 401, {"detail": "Not authenticated"})

        if path == "/jobs" and method == "GET":
            return _FakeResponse(200, [self._public(j) for j in reversed(self.jobs) if j["owner"] == user])
        if path.startswith("/jobs/") and method == "GET":
            jid = urllib.parse.unquote(path[len("/jobs/"):])
            job = next((j for j in self.jobs if j["job_id"] == jid), None)
            if job is None or (job["owner"] != user and not self.leak_jobs):
                raise _http_error(url, 404, {"detail": f"Job not found: {jid}"})
            return _FakeResponse(200, self._public(job))
        if path == "/crates" and method == "POST":
            cid = f"crate_{len(self.crates) + 1}"
            self.crates.append({"crate_id": cid, "name": (body or {}).get("name", ""), "owner": user})
            return _FakeResponse(201, {"crate_id": cid, "name": (body or {}).get("name", "")})
        if path == "/crates" and method == "GET":
            return _FakeResponse(200, {"crates": [{"crate_id": c["crate_id"], "name": c["name"]} for c in self.crates if c["owner"] == user]})
        if path.startswith("/crates/") and path.endswith("/songs") and method == "GET":
            cid = urllib.parse.unquote(path[len("/crates/"):-len("/songs")])
            crate = next((c for c in self.crates if c["crate_id"] == cid), None)
            if crate is None or crate["owner"] != user:
                raise _http_error(url, 404, {"detail": "Crate not found"})
            return _FakeResponse(200, {"songs": []})
        if path.startswith("/crates/") and method == "DELETE":
            cid = urllib.parse.unquote(path[len("/crates/"):])
            crate = next((c for c in self.crates if c["crate_id"] == cid), None)
            if crate is None or crate["owner"] != user:
                raise _http_error(url, 404, {"detail": "Crate not found"})
            self.crates.remove(crate)
            return _FakeResponse(200, {"deleted": True})
        if path == "/library" and method == "GET":
            return _FakeResponse(200, {"stats": {"total": len(LIBRARY_SONGS)}, "songs": [{"name": n} for n in LIBRARY_SONGS]})
        if path == "/analyze" and method == "POST":
            if (body or {}).get("song") not in LIBRARY_SONGS:
                raise _http_error(url, 404, {"detail": "Song not found"})
            return _FakeResponse(202, self._public(self._create_job(user, "analyze")))
        if path == "/dj-remix" and method == "POST":
            return _FakeResponse(202, self._public(self._create_job(user, "dj_remix")))
        if path == "/events/stream" and method == "GET":
            return _FakeStream(self, user)
        raise _http_error(url, 404, {"detail": "Not Found"})


def _run_main(fake: FakeApi, *extra: str):
    argv = ["live_verify", "--base-url", BASE, "--token-a", fake.token_a, "--token-b", fake.token_b, "--skip-render", *extra]
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(lv._OPENER, "open", side_effect=fake.open), \
            mock.patch("urllib.request.urlopen", side_effect=fake.open), \
            mock.patch("sys.argv", argv), redirect_stdout(out), redirect_stderr(err):
        code = lv.main()
    return code, out.getvalue(), err.getvalue()


def _checks(out: str) -> dict:
    return {c["name"]: c for c in json.loads(out)["checks"]}


class B84AuthOffTests(unittest.TestCase):
    def test_auth_off_prints_skip_and_exits_0(self):
        fake = FakeApi(auth_on=False)
        code, out, err = _run_main(fake)
        self.assertIn(SKIP_LINE, out, msg=f"stdout={out!r} stderr={err!r}")
        self.assertEqual(code, 0, msg=f"stdout={out!r} stderr={err!r}")
        self.assertNotIn("LEAK", out)
        # No tenant probe ran: nothing was created as A to be looked up as B.
        self.assertNotIn(("POST", "/crates"), [(m, p) for m, p, _ in fake.calls])


class B84FeaturesTests(unittest.TestCase):
    def test_features_present_passes(self):
        fake = FakeApi()
        code, out, _ = _run_main(fake, "--json")
        checks = _checks(out)
        self.assertIn("features", checks, msg=sorted(checks))
        self.assertTrue(checks["features"]["passed"], checks["features"])
        self.assertIn(("GET", "/features"), [(m, p) for m, p, _ in fake.calls])
        self.assertEqual(code, 0)

    def test_features_missing_fails_the_row(self):
        fake = FakeApi(features_present=False)
        code, out, _ = _run_main(fake, "--json")
        checks = _checks(out)
        self.assertIn("features", checks, msg=sorted(checks))
        self.assertFalse(checks["features"]["passed"])
        self.assertEqual(code, 1)


class B84JobByIdTests(unittest.TestCase):
    def test_leak_fails(self):
        fake = FakeApi(leak_jobs=True)
        code, out, _ = _run_main(fake, "--json")
        checks = _checks(out)
        self.assertIn("job_by_id_isolation", checks, msg=sorted(checks))
        self.assertFalse(checks["job_by_id_isolation"]["passed"])
        self.assertIn("LEAK", checks["job_by_id_isolation"]["detail"])
        self.assertTrue(fake.foreign_job_reads(TOKEN_B), msg=fake.calls)
        self.assertEqual(code, 1)

    def test_isolated_passes(self):
        fake = FakeApi()
        code, out, _ = _run_main(fake, "--json")
        checks = _checks(out)
        self.assertIn("job_by_id_isolation", checks, msg=sorted(checks))
        self.assertTrue(checks["job_by_id_isolation"]["passed"], checks["job_by_id_isolation"])
        # B really asked for one of A's jobs by id (the fake answered 404).
        self.assertTrue(fake.foreign_job_reads(TOKEN_B), msg=fake.calls)
        self.assertEqual(code, 0)


class B84SseTests(unittest.TestCase):
    def test_leak_fails(self):
        fake = FakeApi(leak_sse=True)
        code, out, _ = _run_main(fake, "--json")
        checks = _checks(out)
        self.assertIn("sse_isolation", checks, msg=sorted(checks))
        self.assertFalse(checks["sse_isolation"]["passed"])
        self.assertIn("LEAK", checks["sse_isolation"]["detail"])
        self.assertEqual(code, 1)

    def test_isolated_passes(self):
        fake = FakeApi()
        code, out, _ = _run_main(fake, "--json")
        checks = _checks(out)
        self.assertIn("sse_isolation", checks, msg=sorted(checks))
        self.assertTrue(checks["sse_isolation"]["passed"], checks["sse_isolation"])
        # B was connected and A produced a job event during the window.
        self.assertIn(("GET", "/events/stream", TOKEN_B), fake.calls)
        self.assertTrue(fake.jobs_created_by(USER_A), msg=fake.calls)
        self.assertEqual(code, 0)


class B84DisplayTests(unittest.TestCase):
    def test_job_count_reads_the_array(self):
        fake = FakeApi(jobs_a=3, jobs_b=0)
        code, out, _ = _run_main(fake)
        self.assertIn("Token A auth ok (3 jobs)", out, msg=out)
        self.assertIn("Token B auth ok (0 jobs)", out, msg=out)

    def test_all_rows_present_and_green(self):
        fake = FakeApi()
        code, out, _ = _run_main(fake, "--json")
        data = json.loads(out)
        names = {c["name"] for c in data["checks"]}
        expected = {"health_ready", "auth_token_a", "auth_token_b", "cross_tenant_isolation"} | NEW_CHECKS
        self.assertTrue(expected <= names, msg=f"missing {sorted(expected - names)}")
        self.assertTrue(data["all_passed"], data)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
