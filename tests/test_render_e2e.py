"""
tests/test_render_e2e.py: end to end render checks for the "render" family
(audit plan B3.1, B3.2, and the record_render guard, manifest session and
chain cancel items of B3.5).

Two synthetic tracks live in an isolated library (paths.LIBRARY_DIR is
redirected to a temp dir; the P0 conftest already isolates outputs/ and
data/). Each is a metronome click plus a tonal pad, so the tempo can be read
back with librosa and each song can be located in the mix by the frequency
band of its pad:

    Synth A 120       120 BPM, pad 220 Hz,  60 s
    Synth B 132       132 BPM, pad 1500 Hz, 60 s
    Synth C 126       126 BPM, pad 550 Hz,  60 s
    Synth B long 132  132 BPM, pad 1500 Hz, 240 s   (B3.2 truncation case)

Controls run before the tests were written: librosa.feature.tempo on the
final 15 s reads 120.2 for A, 132.5 for B, 143.6 for B stretched by the
current bpm_b / bpm_a ratio and 120.2 for B stretched by bpm_a / bpm_b. So
the 3 percent window around 120 separates the spec from the defect.

Every render calls the task function directly with a real job id and the
current_user contextvar set, exactly as jobs._wrapper does.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("librosa")

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

from scripts.api import jobs as job_store  # noqa: E402
from scripts.api.auth import DEFAULT_USER, current_user  # noqa: E402
from scripts.api.schemas import JobType  # noqa: E402
from scripts.api.task_modules import remix  # noqa: E402
from scripts.core import paths  # noqa: E402
from scripts.core.config import cfg  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.dj_analysis]

SR = 44100
SONG_A = "Synth A 120"
SONG_B = "Synth B 132"
SONG_C = "Synth C 126"
SONG_B_LONG = "Synth B long 132"
PAD_HZ = {SONG_A: 220.0, SONG_B: 1500.0, SONG_C: 550.0, SONG_B_LONG: 1500.0}
ENGINES = ("current", "classic")


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------

def _synth(bpm: float, seconds: float, pad_hz: float, seed: int) -> np.ndarray:
    """Metronome click (noise burst, accent every 4th beat) over a slowly
    breathing two partial pad. Peak stays under 1.0."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    pad = 0.15 * (np.sin(2 * np.pi * pad_hz * t) + 0.5 * np.sin(2 * np.pi * pad_hz * 2 * t))
    pad *= 0.5 + 0.5 * np.sin(2 * np.pi * t / 8.0) ** 2
    y = pad.astype(np.float32)
    beat = 60.0 / bpm
    click_len = int(0.02 * SR)
    env = np.exp(-np.arange(click_len) / (0.004 * SR)).astype(np.float32)
    burst = rng.standard_normal(click_len).astype(np.float32) * env
    k = 0
    while k * beat < seconds:
        s = int(k * beat * SR)
        e = min(n, s + click_len)
        y[s:e] += (0.9 if k % 4 == 0 else 0.6) * burst[: e - s]
        k += 1
    return np.clip(y, -1.0, 1.0)


def _tail_tempo(y: np.ndarray, sr: int, seconds: float = 15.0) -> float:
    """Tempo over the final `seconds` of the mix (librosa.feature.tempo on
    onset strength), as the spec names it."""
    tail = y[-int(seconds * sr):]
    oenv = librosa.onset.onset_strength(y=tail, sr=sr)
    return float(librosa.feature.tempo(onset_envelope=oenv, sr=sr)[0])


def _dbfs(seg: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64)))) if seg.size else 0.0
    return 20.0 * math.log10(max(rms, 1e-9))


def _dominant_pad_per_second(y: np.ndarray, sr: int, songs: list[str]) -> list[int]:
    """For each 1 s window, the index of the song whose pad band (fundamental
    plus or minus 25 percent, covering a 3 semitone pitch shift) holds the
    most energy."""
    n_fft = 4096
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=sr)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rows = []
    for name in songs:
        hz = PAD_HZ[name]
        m = (f >= hz * 0.75) & (f <= hz * 1.25)
        rows.append(S[m].sum(axis=0))
    return [int(i) for i in np.argmax(np.vstack(rows), axis=0)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory):
    """Isolated library root with the four synthetic songs as full.wav.

    The conftest leaves LIBRARY_DIR on the real catalog; song_dir() reads the
    module global at call time, so redirecting it here isolates every reader
    on the render path (audio_source, credits, stem detection)."""
    root = tmp_path_factory.mktemp("library")
    specs = [
        (SONG_A, 120.0, 60.0, 0),
        (SONG_B, 132.0, 60.0, 1),
        (SONG_C, 126.0, 60.0, 2),
        (SONG_B_LONG, 132.0, 240.0, 3),
    ]
    for name, bpm, secs, seed in specs:
        (root / name).mkdir()
        sf.write(str(root / name / "full.wav"), _synth(bpm, secs, PAD_HZ[name], seed), SR,
                 subtype="PCM_16")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "LIBRARY_DIR", root)
        assert paths.song_dir(SONG_A) == root / SONG_A          # the redirect is live
        from scripts.core.audio_source import resolve_source_file
        assert resolve_source_file(SONG_A) == root / SONG_A / "full.wav"
        yield root


def _new_job(meta: Dict[str, Any]) -> str:
    current_user.set(DEFAULT_USER)
    return job_store.create_job(JobType.DJ_REMIX, meta)


def _read(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32")
    return y, sr


@pytest.fixture(scope="module", params=ENGINES)
def remix_result(request: pytest.FixtureRequest, library: Path) -> Dict[str, Any]:
    """One real task_dj_remix render per engine (A 120 into B 132)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cfg.dj, "mixing_engine", request.param)
        jid = _new_job({"song_a": SONG_A, "song_b": SONG_B})
        res = remix.task_dj_remix(jid, SONG_A, SONG_B, transition_bars=8, preset="club")
    y, sr = _read(res["output"])
    return {"engine": request.param, "res": res, "y": y, "sr": sr, "job_id": jid}


@pytest.fixture(scope="module")
def chain_result(library: Path) -> Dict[str, Any]:
    """One real task_dj_chain render, three songs, on the configured engine."""
    songs = [SONG_A, SONG_B, SONG_C]
    jid = _new_job({"songs": songs})
    res = remix.task_dj_chain(jid, songs, transition_bars=8, preset="club")
    y, sr = _read(res["output"])
    return {"res": res, "y": y, "sr": sr, "songs": songs}


# ---------------------------------------------------------------------------
# B3.1: two song render
# ---------------------------------------------------------------------------

def test_remix_wav_is_44100_pcm24_mono_finite_and_in_range(remix_result):
    res, y = remix_result["res"], remix_result["y"]
    out = Path(res["output"])
    assert out.exists(), res
    info = sf.info(str(out))
    assert (info.samplerate, info.subtype, info.channels) == (44100, "PCM_24", 1)
    assert y.ndim == 1 and len(y) > 0
    assert not np.isnan(y).any()
    assert float(np.abs(y).max()) <= 1.0
    assert res["stream_url"] == f"/outputs/{out.parent.name}/{out.name}"


def test_remix_credits_and_manifest_have_one_credit_per_song(remix_result):
    res = remix_result["res"]
    out_dir = Path(res["output"]).parent
    assert (out_dir / "CREDITS.txt").exists()
    assert (out_dir / "manifest.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["songs"] == [SONG_A, SONG_B]
    assert [c["song"] for c in manifest["credits"]] == [SONG_A, SONG_B]
    assert [c["song"] for c in res["credits"]] == [SONG_A, SONG_B]
    txt = (out_dir / "CREDITS.txt").read_text(encoding="utf-8")
    assert SONG_A in txt and SONG_B in txt


def test_remix_manifest_session_equals_result_session_id(remix_result):
    """B3.5: manifest.session is the result's session_id, not the folder name."""
    res = remix_result["res"]
    manifest = json.loads((Path(res["output"]).parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["session"] == res["session_id"]


def test_remix_stream_url_resolves_through_outputs_route(remix_result, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from scripts.api import main as main_mod

    monkeypatch.delenv("REMIXMATE_AUTH_ENABLED", raising=False)  # acting user "local", as the render
    res = remix_result["res"]
    client = TestClient(main_mod.app)   # routes only, no lifespan
    r = client.get(res["stream_url"])
    assert r.status_code == 200, (r.status_code, r.text[:200])
    assert r.headers["content-type"].startswith("audio/wav")
    assert len(r.content) == Path(res["output"]).stat().st_size


def test_remix_plan_ratio_is_bpm_a_over_bpm_b(remix_result):
    """ARCHITECTURE.md:348: stretch_ratio = bpm_a / bpm_b (B is pulled toward A)."""
    res = remix_result["res"]
    assert res["bpm_a"] == pytest.approx(120.0, rel=0.03)
    assert res["bpm_b"] == pytest.approx(132.0, rel=0.03)
    assert res["tempo_ratio"] == pytest.approx(res["bpm_a"] / res["bpm_b"], rel=0.01)


def test_remix_tail_tempo_is_song_a_tempo(remix_result):
    """The final 15 s of the mix is song B alone, tempo matched to A: 120 BPM
    within 3 percent. On the inverted ratio B lands near 132^2 / 120 = 145."""
    tempo = _tail_tempo(remix_result["y"], remix_result["sr"])
    assert abs(tempo - 120.0) / 120.0 <= 0.03, f"tail tempo {tempo:.1f} BPM ({remix_result['engine']})"


# ---------------------------------------------------------------------------
# B3.1: three song chain
# ---------------------------------------------------------------------------

def test_chain_wav_format_and_three_credits(chain_result):
    res, y = chain_result["res"], chain_result["y"]
    out = Path(res["output"])
    info = sf.info(str(out))
    assert (info.samplerate, info.subtype, info.channels) == (44100, "PCM_24", 1)
    assert not np.isnan(y).any() and float(np.abs(y).max()) <= 1.0
    assert res["n_songs"] == 3 and res["n_transitions"] == 2
    manifest = json.loads((out.parent / "manifest.json").read_text(encoding="utf-8"))
    assert [c["song"] for c in manifest["credits"]] == chain_result["songs"]
    assert (out.parent / "CREDITS.txt").exists()


def test_chain_middle_song_is_not_skipped(chain_result):
    """Each song's pad sits in its own band; the middle song must own a run of
    the mix between the first and the last. The probe on the shipped code
    read A 55 s, B 40 s, C 45 s in that order."""
    dom = _dominant_pad_per_second(chain_result["y"], chain_result["sr"], chain_result["songs"])
    counts = [dom.count(i) for i in range(3)]
    assert counts[1] >= 10, f"middle song owns {counts[1]} s of the mix: {counts}"
    centre = [float(np.median([k for k, d in enumerate(dom) if d == i])) for i in range(3)]
    assert centre[0] < centre[1] < centre[2], (counts, centre)
    # No song may be shorter than one transition window (8 bars at 120 = 16 s)
    # minus a generous margin, and the whole mix must outlast A plus C alone.
    assert chain_result["res"]["duration_sec"] > 60.0 + 60.0 - 2 * 16.0


def test_chain_manifest_session_equals_result_session_id(chain_result):
    res = chain_result["res"]
    manifest = json.loads((Path(res["output"]).parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["session"] == res["session_id"]


def test_chain_stops_between_songs_once_cancelled(library, monkeypatch):
    """B3.5 / B2.5: task_dj_chain polls job_store.is_cancelled between songs.
    A cancel that lands while song 1 loads must stop the chain before song 3
    is touched and before any chain output is written."""
    monkeypatch.setattr(cfg.dj, "mixing_engine", "classic")   # fastest engine; the check is task level
    songs = [SONG_A, SONG_B, SONG_C]
    jid = _new_job({"songs": songs})
    loads: list[str] = []
    real_load = remix.load_source_audio

    def _load_then_cancel(name, *a, **kw):
        out = real_load(name, *a, **kw)
        loads.append(name)
        if len(loads) == 1:
            assert job_store.cancel_job(jid) is True
            assert job_store.is_cancelled(jid) is True
        return out

    monkeypatch.setattr(remix, "load_source_audio", _load_then_cancel)
    out_root = paths.user_outputs_dir()
    before = {p for p in out_root.glob("chain_*")} if out_root.exists() else set()

    try:
        result = remix.task_dj_chain(jid, songs, transition_bars=8, preset="club")
    except Exception as exc:  # noqa: BLE001  a raise is an accepted way to stop
        result = None
        stop = f"raised {type(exc).__name__}: {exc}"
    else:
        stop = f"returned {type(result).__name__}"

    new_dirs = ({p for p in out_root.glob("chain_*")} if out_root.exists() else set()) - before
    written = [w for d in new_dirs for w in d.glob("*.wav")]
    assert len(loads) < 3, f"chain loaded {loads} after the cancel ({stop})"
    assert not written, f"chain wrote {written} after the cancel ({stop})"
    assert not (isinstance(result, dict) and result.get("output")), stop


# ---------------------------------------------------------------------------
# B3.2: max_song_seconds and the fade out on a truncated song B
# ---------------------------------------------------------------------------

def test_max_song_seconds_defaults_to_180():
    assert getattr(cfg.remix, "max_song_seconds", None) == 180


def test_max_song_seconds_env_override(monkeypatch):
    from scripts.core.config import load_config

    monkeypatch.setenv("REMIXMATE_REMIX_MAX_SONG_SECONDS", "45")
    fresh = load_config()
    value = getattr(fresh.remix, "max_song_seconds", None)
    assert isinstance(value, (int, float)) and not isinstance(value, bool), repr(value)
    assert float(value) == 45.0


def test_load_cap_comes_from_config_and_truncation_fades_out(library, monkeypatch):
    """With remix.max_song_seconds at 30 the loader receives 30, the mix is
    short, and the truncated B ends in a fade (last 50 ms under minus 40 dBFS)."""
    monkeypatch.setattr(cfg.dj, "mixing_engine", "classic")
    monkeypatch.setattr(cfg.remix, "max_song_seconds", 30, raising=False)
    durations: list[Any] = []
    real_load = remix.load_source_audio

    def _spy(name, *a, **kw):
        durations.append(kw.get("duration", a[2] if len(a) > 2 else None))
        return real_load(name, *a, **kw)

    monkeypatch.setattr(remix, "load_source_audio", _spy)
    jid = _new_job({"song_a": SONG_A, "song_b": SONG_B})
    res = remix.task_dj_remix(jid, SONG_A, SONG_B, transition_bars=4, preset="club")
    y, sr = _read(res["output"])

    assert durations and all(float(d) == 30.0 for d in durations), durations
    assert len(y) / sr < 60.0, f"mix is {len(y) / sr:.1f} s; both songs should be capped at 30 s"
    assert _dbfs(y[-int(0.05 * sr):]) < -40.0
    assert _dbfs(y[-int(3.05 * sr):-int(0.05 * sr)]) > -40.0


def test_truncated_240s_song_b_ends_with_fade_out(library):
    """Spec case: a 240 s song B under the 180 s cap. Last 50 ms of the mix
    under minus 40 dBFS while the 3 s before are not. The shipped code hard
    cuts: the probe read minus 22.4 dBFS over the last 50 ms."""
    jid = _new_job({"song_a": SONG_A, "song_b": SONG_B_LONG})
    res = remix.task_dj_remix(jid, SONG_A, SONG_B_LONG, transition_bars=8, preset="club")
    y, sr = _read(res["output"])
    last = _dbfs(y[-int(0.05 * sr):])
    before = _dbfs(y[-int(3.05 * sr):-int(0.05 * sr)])
    assert last < -40.0, f"last 50 ms at {last:.1f} dBFS (hard cut)"
    assert before > -40.0, f"the 3 s before the tail sit at {before:.1f} dBFS"


# ---------------------------------------------------------------------------
# B3.5: record_render is guarded
# ---------------------------------------------------------------------------

def test_record_render_failure_is_a_warning_not_a_job_failure(library, monkeypatch, caplog, tmp_path):
    """A data/ write failure inside record_render must not fail a finished
    render. The tripwire is real: history_path's parent is a regular file."""
    from scripts.core import spectral_matching as sm

    monkeypatch.setattr(cfg.dj, "mixing_engine", "classic")
    monkeypatch.setenv("REMIXMATE_SPECTRAL_LEARNING", "1")
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(sm, "_data_dir", lambda: blocker / "data")

    fake_info = {"sims": [0.5] * sm.N_BANDS, "score": 0.5, "entry_bar": 0,
                 "original_entry_bar": 0, "spectral": None, "drop_delta_s": 0.0}
    with pytest.raises(OSError):           # prove the guard has something to guard
        sm.record_render("tripwire", fake_info)
    monkeypatch.setattr(sm, "apply_spectral_matching", lambda *a, **k: dict(fake_info))

    caplog.set_level(logging.WARNING)
    jid = _new_job({"song_a": SONG_A, "song_b": SONG_B})
    res = remix.task_dj_remix(jid, SONG_A, SONG_B, transition_bars=8, preset="club")

    assert Path(res["output"]).exists()
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING
              and ("record_render" in r.getMessage() or "history" in r.getMessage().lower()
                   or "spectral" in r.getMessage().lower())]
    assert warned, [r.getMessage() for r in caplog.records]
