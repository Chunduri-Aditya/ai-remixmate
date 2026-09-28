"""
tests/test_ingest_guards.py — B7.2 ingest guards (spec: plans/audit-the-project-and-sorted-flask.md).

Family "ingest-guards". Every test here runs offline:
  * fetcher tests replay tests/fixtures/ingest/*.json (recorded once with
    ``python -m scripts.download --source <src> "<query>" --search-only``) through a
    fake transport patched over urllib.request.urlopen and requests.Session.request;
    any URL outside the fixtures raises a BaseException tripwire so a leak cannot be
    swallowed by ``except Exception``;
  * ingest tests point LIBRARY_DIR at a temp library, stub the audio download and
    Demucs, and prove the guards by what survives on disk.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import sys
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

import scripts.core.paths as paths
import scripts.curate_catalog as curate
import scripts.download as dl
from scripts.core.license import LicenseInfo, LicenseType, save_license
from scripts.download import CatalogCandidate, IngestResult, ingest_track

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"

CC_BY_URL = "http://creativecommons.org/licenses/by/4.0/"
CC_BY_NC_URL = "http://creativecommons.org/licenses/by-nc/3.0/"


# ---------------------------------------------------------------------------
# Offline transport (fixtures recorded from the real APIs)
# ---------------------------------------------------------------------------

class NetworkTripwire(BaseException):
    """Raised for any URL not in the fixtures. BaseException so no handler eats it."""


def _key(url: str, params: Optional[Dict[str, Any]] = None):
    u = urlsplit(url)
    q = parse_qs(u.query, keep_blank_values=True)
    for k, v in (params or {}).items():
        vals = v if isinstance(v, (list, tuple)) else [v]
        q.setdefault(k, []).extend(str(x) for x in vals)
    return (u.netloc, u.path, tuple(sorted((k, tuple(sorted(v))) for k, v in q.items())))


def _load_fixture(name: str) -> Dict[str, Any]:
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


class _FakeRequestsResponse:
    def __init__(self, rec: Dict[str, Any]) -> None:
        self._rec = rec
        self.status_code = rec["status"]
        self.url = rec["final_url"]
        self.headers = {"content-type": rec.get("content_type") or ""}

    def json(self) -> Any:
        return self._rec["json"]

    @property
    def text(self) -> str:
        return json.dumps(self._rec["json"])

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8")

    def raise_for_status(self) -> None:
        return None

    def __enter__(self) -> "_FakeRequestsResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _FakeUrlopenResponse:
    def __init__(self, rec: Dict[str, Any]) -> None:
        self._body = json.dumps(rec["json"]).encode("utf-8")
        self.status = rec["status"]
        self.headers = {"Content-Type": rec.get("content_type") or ""}

    def read(self, *_: Any) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_FakeUrlopenResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class OfflineHTTP:
    """Serves recorded responses; records every call so a test can prove it was used."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._map = {_key(r["final_url"]): r for r in records}
        self.calls: List[str] = []

    def _lookup(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.calls.append(url)
        rec = self._map.get(_key(url, params))
        if rec is None:
            raise NetworkTripwire(f"network call outside fixtures: {url} params={params}")
        return rec

    def requests_request(self, _session: Any, method: str, url: str, **kw: Any) -> Any:
        return _FakeRequestsResponse(self._lookup(url, kw.get("params")))

    def urlopen(self, url_or_req: Any, *a: Any, **kw: Any) -> Any:
        url = getattr(url_or_req, "full_url", url_or_req)
        return _FakeUrlopenResponse(self._lookup(str(url)))


def _install_offline(monkeypatch: pytest.MonkeyPatch, fixture: str) -> OfflineHTTP:
    http = OfflineHTTP(_load_fixture(fixture)["responses"])
    monkeypatch.setattr(urllib.request, "urlopen", http.urlopen)
    monkeypatch.setattr(dl, "urlopen", http.urlopen, raising=False)
    import requests
    def _session_request(session: Any, method: str, url: str, **kw: Any) -> Any:
        return http.requests_request(session, method, url, **kw)
    monkeypatch.setattr(requests.Session, "request", _session_request)
    return http


# ---------------------------------------------------------------------------
# Fetcher parsing against the recorded fixtures
# ---------------------------------------------------------------------------

def test_ccmixter_fixture_parses_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    http = _install_offline(monkeypatch, "ccmixter_search.json")
    cands = dl.search_catalog("ccmixter", "glitch hop", limit=3)
    assert http.calls, "fetcher made no HTTP call through the patched transport"
    assert len(cands) == 3
    first = cands[0]
    assert first.source == "ccmixter"
    assert first.title == "Down Like Rain"
    assert first.artist == "Scott Altham"
    assert unquote(first.audio_url) == (
        "https://ccmixter.org/content/scottaltham/scottaltham_-_Down_Like_Rain.mp3"
    )
    assert first.license_url == CC_BY_NC_URL
    assert first.license_str == "Attribution Noncommercial  (3.0)"
    assert first.suggested_name() == "Scott Altham - Down Like Rain"
    assert [c.artist for c in cands] == ["Scott Altham", "Kaer Trouz", "cdk"]
    # the live seed run must show an NC candidate rejected: this fixture supplies it
    info = dl.classify_license(first.source, license_url=first.license_url, license_str=first.license_str)
    assert info.license_type is LicenseType.CC_BY_NC
    assert dl.is_publishable(info) is False


def test_internet_archive_fixture_parses_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    http = _install_offline(monkeypatch, "archive_search.json")
    cands = dl.search_catalog("archive", "netlabel ambient", limit=2)
    assert len(http.calls) >= 3, f"expected search + 2 metadata calls, got {http.calls}"
    assert len(cands) == 2
    first, second = cands
    assert first.source == "archive"
    assert first.title == "_"
    assert first.artist == "Lost Elements Netlabel"
    assert unquote(first.audio_url) == (
        "https://archive.org/download/le004PolarAviation-MemoriesEp/PolarAviation-MemoriesOfTheSea.mp3"
    )
    assert first.license_url == "http://creativecommons.org/licenses/by-nc-nd/3.0/"
    assert first.source_url == "https://archive.org/details/le004PolarAviation-MemoriesEp"
    assert second.title == "Objective Tinnitus Live Part 6 GHGR 020116"
    assert second.artist == "Exomène"
    assert unquote(second.audio_url) == (
        "https://archive.org/download/GHGR020116/"
        "Objective Tinnitus - Part 6 GHGR 020116 - 01 Part 6.mp3"
    )
    assert second.license_url == "http://creativecommons.org/licenses/by-nc-nd/3.0/"


def test_offline_transport_trips_on_unknown_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the fake transport must refuse anything not recorded."""
    _install_offline(monkeypatch, "ccmixter_search.json")
    with pytest.raises(NetworkTripwire):
        dl.search_catalog("ccmixter", "a query that was never recorded", limit=3)


# ---------------------------------------------------------------------------
# Ingest guards on a temp library
# ---------------------------------------------------------------------------

def _cand(title: str = "Open Track", artist: Optional[str] = "Open Artist",
          license_url: str = CC_BY_URL, source: str = "ccmixter") -> CatalogCandidate:
    return CatalogCandidate(
        title=title, artist=artist, source=source,
        audio_url=f"https://{source}.example/{uuid.uuid4().hex}.mp3",
        license_url=license_url,
    )


def _reject_or_result(call) -> Optional[IngestResult]:
    """A guard may raise ValueError or return success=False; both count as rejection."""
    try:
        res = call()
    except ValueError:
        return None
    assert res.success is False, f"expected rejection, got success: {res}"
    return res


@pytest.fixture
def temp_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(paths, "LIBRARY_DIR", lib)
    monkeypatch.setattr(dl, "LIBRARY_DIR", lib, raising=False)
    download = MagicMock(side_effect=RuntimeError("connection refused"))
    separate = MagicMock(return_value={})
    monkeypatch.setattr(dl, "_download_audio", download)
    monkeypatch.setattr(dl, "_separate_stems", separate)
    monkeypatch.setattr(dl, "get_library_manager", MagicMock())
    return {"lib": lib, "root": tmp_path, "download": download, "separate": separate}


def _fake_download_writes_wav(url: str, dest_dir: Path, progress_cb: Any = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    wav = dest_dir / "full.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")
    return wav


def test_empty_sanitized_name_rejected_before_any_path(temp_library: Dict[str, Any]) -> None:
    """Title '???' sanitises to '' so song_dir('') IS the library root."""
    lib = temp_library["lib"]
    keep = lib / "Keep Me"
    keep.mkdir()
    (keep / "full.wav").write_bytes(b"keep")

    cand = _cand(title="???", artist=None)
    assert cand.suggested_name() == "", "precondition: the name must sanitise to empty"

    res = _reject_or_result(lambda: ingest_track(cand))
    assert lib.is_dir(), "library root was removed by the failed download"
    assert (keep / "full.wav").read_bytes() == b"keep", "sibling track lost"
    assert temp_library["download"].call_count == 0, "download ran for an empty name"
    assert not (lib / "license.json").exists(), "licence written into the library root"
    if res is not None:
        assert res.error


@pytest.mark.parametrize("bad_name", ["", ".", "../escape", "nested/child", "/abs/elsewhere"])
def test_dest_dir_must_be_direct_child_of_library(
    temp_library: Dict[str, Any], bad_name: str
) -> None:
    lib, root = temp_library["lib"], temp_library["root"]
    before = sorted(p.relative_to(root) for p in root.rglob("*"))

    _reject_or_result(lambda: ingest_track(_cand(), name=bad_name))

    assert temp_library["download"].call_count == 0, f"download ran for name {bad_name!r}"
    after = sorted(p.relative_to(root) for p in root.rglob("*"))
    assert after == before, f"filesystem changed for name {bad_name!r}: {after}"
    assert lib.is_dir()
    assert not (root / "escape").exists()


def test_library_root_survives_failed_download_of_normal_track(
    temp_library: Dict[str, Any],
) -> None:
    lib = temp_library["lib"]
    keep = lib / "Keep Me"
    keep.mkdir()
    (keep / "full.wav").write_bytes(b"keep")

    res = ingest_track(_cand(title="Doomed"), separate=False)

    assert res.success is False
    assert "download" in (res.error or "").lower()
    assert not (lib / "Open Artist - Doomed").exists(), "partial dir left behind"
    assert (keep / "full.wav").read_bytes() == b"keep"
    assert lib.is_dir()


def test_demucs_failure_returns_failure_and_removes_partial_dir(
    temp_library: Dict[str, Any],
) -> None:
    lib = temp_library["lib"]
    temp_library["download"].side_effect = _fake_download_writes_wav
    temp_library["separate"].return_value = {}          # Demucs produced nothing

    res = ingest_track(_cand(title="Stemless"), separate=True, auto_analyze=False)

    assert temp_library["separate"].call_count == 1
    assert res.success is False, "Demucs failure reported as success"
    assert res.error, "no error message on Demucs failure"
    assert not (lib / "Open Artist - Stemless").exists(), "partial dir kept without --keep-partial"
    assert lib.is_dir()


def test_demucs_failure_keep_partial_keeps_dir(temp_library: Dict[str, Any]) -> None:
    lib = temp_library["lib"]
    params = inspect.signature(ingest_track).parameters
    assert "keep_partial" in params, f"ingest_track has no keep_partial: {list(params)}"
    temp_library["download"].side_effect = _fake_download_writes_wav
    temp_library["separate"].return_value = {}

    res = ingest_track(_cand(title="Stemless"), separate=True, auto_analyze=False, keep_partial=True)

    assert res.success is False
    assert (lib / "Open Artist - Stemless" / "full.wav").exists(), "partial dir removed despite keep_partial"


def test_no_split_ingest_still_succeeds(temp_library: Dict[str, Any]) -> None:
    """Allow pair for the Demucs guard: --no-split never runs Demucs and stays a success."""
    lib = temp_library["lib"]
    temp_library["download"].side_effect = _fake_download_writes_wav

    res = ingest_track(_cand(title="Raw"), separate=False)

    assert res.success is True
    assert temp_library["separate"].call_count == 0
    assert (lib / "Open Artist - Raw" / "full.wav").exists()
    assert (lib / "Open Artist - Raw" / "license.json").exists()


def test_cli_has_keep_partial_flag() -> None:
    dests = {a.dest for a in dl._build_parser()._actions}
    assert "keep_partial" in dests, f"--keep-partial missing; parser dests: {sorted(dests)}"


# --- "already in library" shortcut runs after classification and re checks disk ---

def _seed_existing(lib: Path, name: str, on_disk: LicenseType) -> Path:
    d = lib / name
    d.mkdir()
    (d / "full.wav").write_bytes(b"existing")
    (d / "vocals.wav").write_bytes(b"v")
    save_license(d, LicenseInfo(license_type=on_disk, source="ccmixter"))
    return d


def test_shortcut_gate_rejects_nc_candidate_already_on_disk(
    temp_library: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REMIXMATE_LICENSE_ENFORCE", "1")
    lib = temp_library["lib"]
    _seed_existing(lib, "Scott Altham - Down Like Rain", LicenseType.CC_BY_NC)
    cand = _cand(title="Down Like Rain", artist="Scott Altham", license_url=CC_BY_NC_URL)

    res = ingest_track(cand)

    assert res.success is False, "NC track already on disk was returned as a success"
    assert res.rejected_license == LicenseType.CC_BY_NC.value
    assert temp_library["download"].call_count == 0


def test_shortcut_rechecks_on_disk_licence_against_gate(
    temp_library: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate metadata says CC BY, but license.json on disk says NC: the disk wins."""
    monkeypatch.setenv("REMIXMATE_LICENSE_ENFORCE", "1")
    lib = temp_library["lib"]
    _seed_existing(lib, "Open Artist - Open Track", LicenseType.CC_BY_NC)

    res = ingest_track(_cand(license_url=CC_BY_URL))

    assert res.success is False, "on disk NC licence bypassed the gate through the shortcut"
    assert temp_library["download"].call_count == 0


def test_shortcut_allows_publishable_existing_track(
    temp_library: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow pair: CC BY on disk and in metadata short circuits without a download."""
    monkeypatch.setenv("REMIXMATE_LICENSE_ENFORCE", "1")
    lib = temp_library["lib"]
    _seed_existing(lib, "Open Artist - Open Track", LicenseType.CC_BY)

    res = ingest_track(_cand(license_url=CC_BY_URL))

    assert res.success is True
    assert temp_library["download"].call_count == 0
    assert res.license is not None and res.license.license_type is LicenseType.CC_BY
    assert "vocals" in res.stems


# --- --dry-run: search, classify, gate, print, no download ---

def test_cli_dry_run_searches_classifies_gates_prints_without_download(
    temp_library: Dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
) -> None:
    dests = {a.dest for a in dl._build_parser()._actions}
    assert "dry_run" in dests, f"--dry-run missing; parser dests: {sorted(dests)}"

    monkeypatch.setenv("REMIXMATE_LICENSE_ENFORCE", "1")
    monkeypatch.setattr("shutil.which", lambda name: "/opt/fake/ffmpeg")
    cands = [
        _cand(title="Down Like Rain", artist="Scott Altham", license_url=CC_BY_NC_URL),
        _cand(title="Open Track", artist="Open Artist", license_url=CC_BY_URL),
    ]
    search = MagicMock(return_value=cands)
    monkeypatch.setattr(dl, "search_catalog", search)
    monkeypatch.setattr(sys, "argv", ["scripts.download", "--source", "ccmixter", "glitch hop", "--dry-run"])
    caplog.set_level(logging.INFO)

    rc = dl.main()

    out = capsys.readouterr()
    text = out.out + out.err + caplog.text
    assert rc == 0
    assert search.call_count == 1
    assert temp_library["download"].call_count == 0, "dry run downloaded audio"
    assert temp_library["separate"].call_count == 0, "dry run ran Demucs"
    assert list(temp_library["lib"].iterdir()) == [], "dry run wrote into the library"
    assert "Down Like Rain" in text and "Open Track" in text
    nc_lines = [ln for ln in text.splitlines() if "Down Like Rain" in ln]
    assert any(LicenseType.CC_BY_NC.value in ln for ln in nc_lines), nc_lines
    ok_lines = [ln for ln in text.splitlines() if "Open Track" in ln]
    assert any(re.search(r"\bcc_by\b", ln) for ln in ok_lines), ok_lines
    assert re.search(r"(?i)reject|not publishable|not permitted|✗|skip", text), "no gate verdict printed"


# ---------------------------------------------------------------------------
# curate_catalog: state under DATA_DIR, done only with a success, estimate
# ---------------------------------------------------------------------------

def _manifest(tmp_path: Path, entries: List[Dict[str, Any]]) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(entries), encoding="utf-8")
    return p


@pytest.fixture
def curate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    # cwd moves to tmp so a relative "data/..." write is visible and never hits the repo
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REMIXMATE_LICENSE_ENFORCE", "1")
    assert paths.DATA_DIR != paths.PROJECT_ROOT / "data", "tests must not use the real data/ tree"
    search = MagicMock()
    ingest = MagicMock()
    monkeypatch.setattr(curate, "search_catalog", search)
    monkeypatch.setattr(curate, "ingest_track", ingest)
    return {"tmp": tmp_path, "search": search, "ingest": ingest}


def test_curate_state_file_lives_under_data_dir(curate_env: Dict[str, Any]) -> None:
    query = f"state-{uuid.uuid4().hex}"
    manifest = _manifest(curate_env["tmp"], [{"source": "ccmixter", "query": query, "pick": 1}])
    curate_env["search"].return_value = [_cand(title="Open Track")]
    curate_env["ingest"].return_value = IngestResult(name="Open Artist - Open Track", success=True)

    rc = curate.curate_catalog(manifest, dry_run=False)

    assert rc == 0
    assert not (curate_env["tmp"] / "data").exists(), "state written relative to cwd, not DATA_DIR"
    key = f"ccmixter:{query}"
    hits = [p for p in paths.DATA_DIR.rglob("*.json") if key in p.read_text(encoding="utf-8")]
    assert hits, f"no state file under DATA_DIR={paths.DATA_DIR} records {key!r}"


def test_curate_entry_not_done_after_zero_successes(curate_env: Dict[str, Any]) -> None:
    query = f"fail-{uuid.uuid4().hex}"
    manifest = _manifest(curate_env["tmp"], [{"source": "ccmixter", "query": query, "pick": 1}])
    curate_env["search"].return_value = [_cand(title="Open Track")]
    curate_env["ingest"].return_value = IngestResult(
        name="Open Artist - Open Track", success=False, error="download failed: boom"
    )

    assert curate.curate_catalog(manifest, dry_run=False) == 1
    assert curate_env["search"].call_count == 1
    assert curate.curate_catalog(manifest, dry_run=False) == 1
    assert curate_env["search"].call_count == 2, "entry with zero successes was marked done and skipped"


def test_curate_entry_done_after_one_success(curate_env: Dict[str, Any]) -> None:
    """Allow pair: a success marks the entry done and the rerun skips it."""
    query = f"ok-{uuid.uuid4().hex}"
    manifest = _manifest(curate_env["tmp"], [{"source": "ccmixter", "query": query, "pick": 1}])
    curate_env["search"].return_value = [_cand(title="Open Track")]
    curate_env["ingest"].return_value = IngestResult(name="Open Artist - Open Track", success=True)

    assert curate.curate_catalog(manifest, dry_run=False) == 0
    curate.curate_catalog(manifest, dry_run=False)   # exit code of a fully resumed run is unspecified
    assert curate_env["search"].call_count == 1, "completed entry was searched again"


def test_curate_estimate_uses_min_pick_publishable(
    curate_env: Dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    manifest = _manifest(curate_env["tmp"], [{"source": "ccmixter", "query": "three cc by", "pick": 1}])
    curate_env["search"].return_value = [_cand(title=f"Track {i}") for i in range(3)]

    rc = curate.curate_catalog(manifest, dry_run=True)

    assert rc == 0
    curate_env["ingest"].assert_not_called()
    # the line reads "Estimated storage (0.2 GB per track with stems): <total> GB"; take the total
    m = re.search(r"Estimated storage[^\n]*:\s*(\d+\.\d+) GB", caplog.text)
    assert m, caplog.text
    # pick=1, publishable=3 -> min(1, 3) tracks * 0.2 GB
    assert m.group(1) == "0.2", f"estimate counted all publishable, not min(pick, publishable): {m.group(0)}"
