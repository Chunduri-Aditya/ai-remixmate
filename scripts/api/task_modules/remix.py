"""
scripts/api/task_modules/remix.py — DJ remix and chain task functions.

task_dj_remix          — render a 2-song DJ transition mix (full output)
task_dj_chain          — render an N-song continuous DJ mix
task_remix_preview     — render only the transition window for fast audition
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from scripts.api.jobs import update_job
from scripts.core.audio_source import load_source_audio
from scripts.core.audit import log_audit
from scripts.core.license import load_license
from scripts.core.paths import song_dir, user_outputs_dir

log = logging.getLogger(__name__)


class MixEngine(NamedTuple):
    """One render path: analysis, planning, rendering, and mastering together.

    The four callables must come from the same engine generation — mixing a
    classic plan with the current renderer (or vice versa) is unsupported.
    """
    analyze: Callable            # audio, sr → SongStructure
    plan_transition: Callable    # struct_a, struct_b, … → TransitionPlan
    DJEngine: type
    master_mix: Callable
    dj_target_lufs: float        # LUFS target for full remix renders
    name: str                    # "classic" | "current"


def _resolve_engine() -> MixEngine:
    """Pick the transition engine per config `dj.mixing_engine`.

    "classic" (default) — the vendored AI_REMIXMATE_BEST_MIXING_ALGO path
    (scripts/core/classic_engine/), mastered at −8 LUFS like the reference
    renders that path was judged on. "current" — the live engine
    (scripts/core/dj_engine.py) with its config-driven loudness target
    (audio.target_lufs, default −14). Resolved per job so a config change
    takes effect on API restart without touching code.
    """
    try:
        from scripts.core.config import cfg
        name = str(getattr(getattr(cfg, "dj", None), "mixing_engine", "classic")).lower()
    except Exception:
        name = "classic"
    if name not in ("classic", "current"):
        name = "classic"

    if name == "classic":
        from scripts.core.classic_engine import (
            CLASSIC_DJ_TARGET_LUFS,
            DJEngine,
            _analyze_impl,
            master_mix,
            plan_transition,
        )
        return MixEngine(_analyze_impl, plan_transition, DJEngine,
                         master_mix, CLASSIC_DJ_TARGET_LUFS, name)

    from scripts.core.dj_engine import DJEngine, _analyze_impl, plan_transition
    from scripts.core.mastering import master_mix
    # Loudness target comes from config (audio.target_lufs, default −14 —
    # the Spotify/Apple normalization level). The old hardcoded −8 "club
    # loud" target forced ~+10 dB of gain into the true-peak limiter,
    # crushing dynamics for zero audible benefit on streaming playback,
    # which turns everything back down to ~−14 anyway. Set
    # audio.target_lufs: -8.0 in config.local.yaml for club/PA use.
    try:
        from scripts.core.config import cfg
        target_lufs = float(getattr(getattr(cfg, "audio", None), "target_lufs", -14.0))
    except Exception:
        target_lufs = -14.0
    return MixEngine(_analyze_impl, plan_transition, DJEngine,
                     master_mix, target_lufs, name)


def _transition_timeline(plan, struct_b) -> Dict[str, Any]:
    """Key transition moments in OUTPUT seconds, for UI timelines.

    B-local times map to the output via the stretch ratio: B is tempo-matched
    to A, so an event at B-local time t lands at
    exit_time_a + (t − entry_time_b) / tempo_shift_ratio.
    """
    exit_s = float(plan.exit_time_a)
    trans_s = float(plan.transition_seconds)
    bars = max(int(plan.transition_bars), 1)
    swap_bar = int(getattr(getattr(plan, "eq", None), "bass_swap_bar", bars // 2) or bars // 2)

    ratio = float(plan.tempo_shift_ratio or 1.0) or 1.0
    drop_b = getattr(struct_b, "drop_position", None)
    drop_out = None
    if drop_b is not None and float(drop_b) >= float(plan.entry_time_b):
        drop_out = exit_s + (float(drop_b) - float(plan.entry_time_b)) / ratio

    return {
        "overlap_start_s": round(exit_s, 2),
        "bass_swap_s":     round(exit_s + swap_bar * (trans_s / bars), 2),
        "overlap_end_s":   round(exit_s + trans_s, 2),
        "drop_b_s":        round(drop_out, 2) if drop_out is not None else None,
    }


def _should_use_stem_blend(
    has_stems_a: bool,
    has_stems_b: bool,
    harmonic_score: float,
    spectral_score: Optional[float],
    threshold: float,
) -> Tuple[bool, Optional[float]]:
    """Decide the render path: full stem-aware blend, or the plain crossfade.

    DJ_ENGINE_IMPROVEMENT_PLAN.md §2.1 — before this, the only gate on
    render_stem_blend() vs. render() was whether stem files exist on disk;
    two tracks that don't actually suit a stem blend (harmonic clash, no
    usable spectral cue match) still got the full pipeline forced onto them.
    This adds compatibility as a second, independent condition, using signals
    plan_transition()/apply_spectral_matching() already computed — it never
    invents a third render path, "low confidence" just means the existing
    has_stems_a-and-has_stems_b path stays False, exactly as if stems were
    simply missing.

    Returns (use_stem_blend, transition_confidence). transition_confidence is
    None when the gate didn't evaluate (stems missing, so it's moot; or
    harmonic_score is the -1.0 "not computed" sentinel, e.g. some classic-
    engine paths) — missing data is never treated as evidence of a bad
    pairing, the same "0.0/unknown == don't penalize" convention used
    throughout this codebase (compatibility_score(), etc.).
    """
    if not (has_stems_a and has_stems_b):
        return False, None

    if harmonic_score is None or harmonic_score < 0.0:
        return True, None

    if spectral_score is not None:
        confidence = 0.5 * harmonic_score + 0.5 * spectral_score
    else:
        confidence = harmonic_score

    return confidence >= threshold, confidence


# ---------------------------------------------------------------------------
# Mix credits — CREDITS.txt + manifest.json next to every rendered mix
# ---------------------------------------------------------------------------

def _mix_credits(song_names: list) -> list:
    """One credit entry per source track, read from its library `license.json`.

    A track with no readable licence still gets an entry (`license "unknown"`,
    `artist None`) — an attribution obligation is never silently dropped from a
    rendered derivative because a file was missing. `load_license` returns None
    on any read error (and logs it itself), so this cannot raise.
    """
    credits: list = []
    for name in song_names:
        info = load_license(song_dir(name))
        if info is None:
            credits.append({
                "song": name,
                "artist": None,
                "license": "unknown",
                "license_url": None,
                "source": None,
                "source_url": None,
                "attribution_required": True,   # conservative when unknown
            })
        else:
            credits.append({
                "song": name,
                "artist": info.artist_name,
                "license": info.license_type.value,
                "license_url": info.license_url,
                "source": info.source,
                "source_url": info.source_url,
                "attribution_required": info.attribution_required,
            })
    return credits


def _write_mix_credits(out_dir: Path, song_names: list) -> list:
    """Write `CREDITS.txt` + `manifest.json` into `out_dir`; return the credits.

    Never raises — a credits-write failure must not fail an otherwise-good
    render. The returned list also goes into the task response so the client
    can show attribution without re-reading the files.
    """
    credits = _mix_credits(song_names)
    try:
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        lines = [
            "AI RemixMate — mix credits",
            f"Generated {generated}",
            "",
            "This mix is a derivative work built from the tracks below. Creative",
            "Commons BY tracks must credit the artist wherever you share the mix.",
            "",
        ]
        for i, c in enumerate(credits, 1):
            lic = "unknown" if c["license"] == "unknown" else c["license"].upper().replace("_", "-")
            lines.append(f"{i}. {c['song']}")
            lines.append(f"   Artist:  {c['artist'] or 'unknown'}")
            lines.append(f"   License: {lic}" + (f"  ({c['license_url']})" if c["license_url"] else ""))
            lines.append(f"   Source:  {c['source'] or 'unknown'}" + (f"  {c['source_url']}" if c["source_url"] else ""))
            lines.append("")

        (out_dir / "CREDITS.txt").write_text("\n".join(lines), encoding="utf-8")
        (out_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "session": out_dir.name,
                    "generated_utc": generated,
                    "songs": list(song_names),
                    "credits": credits,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:   # non-critical: log, keep the render
        log.warning("credits: could not write CREDITS/manifest into %s: %s", out_dir.name, e)
    return credits


def task_dj_remix(
    job_id: str,
    song_a: str,
    song_b: str,
    transition_bars: int,
    preset: str,
    bridge_beat_mode: str = "none",
    bridge_beat_genre: str = "auto",
    bridge_beat_intensity: float = 0.38,
    bridge_beat_path: Optional[str] = None,
    transition_effect: str = "auto",
) -> Dict[str, Any]:
    log_audit("dj_remix_start", resource=f"{song_a} → {song_b}", job_id=job_id,
              metadata={"transition_bars": transition_bars, "preset": preset, "effect": transition_effect})
    update_job(job_id, progress=0.05, message="Loading audio…")

    eng = _resolve_engine()

    # Resolve source audio per song: full.wav → full_enhanced.wav → stem sum.
    # Raises FileNotFoundError(song name) if the song has no usable audio.
    sr = 44100
    audio_a, _ = load_source_audio(song_a, sr=sr, mono=True, duration=180.0)
    audio_b, _ = load_source_audio(song_b, sr=sr, mono=True, duration=180.0)

    update_job(job_id, progress=0.25, message="Analysing structure…")

    struct_a = eng.analyze(audio_a, sr)
    struct_b = eng.analyze(audio_b, sr)

    update_job(job_id, progress=0.45, message="Planning transition…")

    plan = eng.plan_transition(struct_a, struct_b, transition_bars=transition_bars)

    # Learned spectrogram matching — may move B's entry cue to the phrase
    # whose spectrum best fits A's outro (dj.spectral_matching; band weights
    # train on 👍/👎 via POST /dj-remix/feedback). None → plan untouched.
    from scripts.core.spectral_matching import apply_spectral_matching, record_render
    spectral_info = apply_spectral_matching(plan, audio_a, audio_b, struct_b, sr)

    # ── Bridge beat ──────────────────────────────────────────────────────────
    bridge_audio: Optional[np.ndarray] = None
    bridge_label = "none"

    if bridge_beat_mode == "auto":
        update_job(job_id, progress=0.52, message="Synthesising bridge beat…")
        try:
            from scripts.core.beat_synth import render_beat
            # Render enough bars to fill (and loop across) the transition window
            beat_bars = max(4, transition_bars)
            bridge_audio = render_beat(
                bpm=plan.bpm_a,
                genre=bridge_beat_genre,
                bars=beat_bars,
                sr=sr,
                intensity=bridge_beat_intensity,
                build_up=True,
            )
            bridge_label = f"auto:{bridge_beat_genre}"
        except Exception as exc:
            # Non-critical — proceed without the beat
            bridge_audio = None
            bridge_label = f"auto:failed({exc})"

    elif bridge_beat_mode == "file" and bridge_beat_path:
        update_job(job_id, progress=0.52, message="Loading bridge beat file…")
        try:
            bridge_audio, beat_sr = librosa.load(
                bridge_beat_path, sr=sr, mono=True
            )
            bridge_label = f"file:{Path(bridge_beat_path).name}"
        except Exception as exc:
            bridge_audio = None
            bridge_label = f"file:failed({exc})"

    # ── Detect stems for intelligent per-stem mixing ──────────────────────────
    stems_dir_a = song_dir(song_a)
    stems_dir_b = song_dir(song_b)
    has_stems_a = any(
        (stems_dir_a / f"{s}.wav").exists() or (stems_dir_a / f"{s}.flac").exists()
        for s in ("vocals", "drums", "bass", "other")
    )
    has_stems_b = any(
        (stems_dir_b / f"{s}.wav").exists() or (stems_dir_b / f"{s}.flac").exists()
        for s in ("vocals", "drums", "bass", "other")
    )
    from scripts.core.config import cfg as _cfg
    _confidence_threshold = float(
        getattr(getattr(_cfg, "dj", None), "transition_confidence_threshold", 0.30)
    )
    use_stem_blend, transition_confidence = _should_use_stem_blend(
        has_stems_a, has_stems_b,
        plan.harmonic_score,
        spectral_info.get("score") if spectral_info else None,
        _confidence_threshold,
    )
    if has_stems_a and has_stems_b and not use_stem_blend:
        log.info(
            "Transition confidence %.3f < %.2f threshold (%s → %s) — "
            "falling back to plain crossfade instead of stem blend",
            transition_confidence, _confidence_threshold, song_a, song_b,
        )

    if use_stem_blend:
        update_job(job_id, progress=0.55,
                   message="Rendering stem-aware DJ mix (intelligent instrument blending)…")
    else:
        update_job(job_id, progress=0.55, message="Rendering DJ mix…")

    engine = eng.DJEngine(sr=sr)

    if use_stem_blend:
        mix = engine.render_stem_blend(
            audio_a, audio_b, plan,
            stems_dir_a=stems_dir_a,
            stems_dir_b=stems_dir_b,
            full_output=True,
            bridge_beat=bridge_audio,
            bridge_gain=bridge_beat_intensity,
            transition_effect=transition_effect,
        )
    else:
        mix = engine.render(
            audio_a, audio_b, plan,
            full_output=True,
            bridge_beat=bridge_audio,
            bridge_gain=bridge_beat_intensity,
            transition_effect=transition_effect,
        )

    # Dynamic spectral sync — time-varying multiband EQ over the overlap
    # window easing the mix's spectrum from A's outro to B's profile
    # (dj.spectral_sync; bass band stays the engine's authority). Runs on the
    # rendered output, so both engines get it without modification.
    update_job(job_id, progress=0.78, message="Syncing transition spectrum…")
    from scripts.core.spectral_sync import apply_spectral_sync
    mix, sync_info = apply_spectral_sync(mix, sr, plan)

    update_job(job_id, progress=0.80, message="Mastering output…")

    # Engine-matched mastering: classic → vendored chain at −8 LUFS (the
    # reference-render sound); current → live chain at audio.target_lufs
    # (default −14, the streaming normalization level — see _resolve_engine).
    mix_mastered, quality_report = eng.master_mix(mix, sr=sr, target_lufs=eng.dj_target_lufs)

    update_job(job_id, progress=0.90, message="Saving output…")

    session_id = str(uuid.uuid4())[:8]
    out_dir = user_outputs_dir() / f"dj_{session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dj_{song_a[:20]}_{song_b[:20]}.wav"

    sf.write(str(out_path), mix_mastered, sr, subtype="PCM_24")

    credits = _write_mix_credits(out_dir, [song_a, song_b])

    if spectral_info:
        record_render(session_id, spectral_info)

    # Compute AI transition score if music intelligence features are available
    ts_data: dict = {}
    try:
        from scripts.core.music_intelligence import compute_track_vector, compute_transition_score
        # analyze_structure() already computed both MusicVectors and stashed
        # them on the structures — recomputing here from raw audio cost ~5 s
        # per render and could disagree with the key the plan was built on.
        # getattr: the classic engine's SongStructure predates the
        # music_vector field, so classic renders recompute here.
        vec_a = getattr(struct_a, "music_vector", None) or compute_track_vector(audio_a, sr, bpm=struct_a.bpm)
        vec_b = getattr(struct_b, "music_vector", None) or compute_track_vector(audio_b, sr, bpm=struct_b.bpm)
        ts    = compute_transition_score(vec_a, vec_b)
        ts_data = {
            "ts_overall":       round(ts.overall, 3),
            "ts_beat":          round(ts.beat_alignment, 3),
            "ts_harmonic":      round(ts.harmonic_match, 3),
            "ts_energy":        round(ts.energy_smoothness, 3),
            "ts_vocal_clash":   round(ts.vocal_clash, 3),
            "ts_notes":         ts.notes,
            "ts_recommended_bars": ts.recommended_transition_bars,
        }
    except Exception:
        pass   # non-critical

    log_audit("dj_remix_complete", resource=str(out_path), job_id=job_id,
              metadata={"song_a": song_a, "song_b": song_b, "stem_blend": use_stem_blend})
    return {
        "output":            str(out_path),
        "session_id":        session_id,
        # Browser-playable URL — served by GET /outputs/{session_id}/{filename}.
        # NOTE: out_dir is named "dj_{session_id}", not bare session_id, so the
        # path segment here must match out_dir.name, not the raw session_id.
        "stream_url":        f"/outputs/{out_dir.name}/{out_path.name}",
        "engine":            eng.name,
        "song_a":            song_a,
        "song_b":            song_b,
        "bpm_a":             round(plan.bpm_a, 1),
        "bpm_b":             round(plan.bpm_b, 1),
        "transition_bars":   plan.transition_bars,
        "transition_sec":    round(plan.transition_seconds, 1),
        "tempo_ratio":       round(plan.tempo_shift_ratio, 4),
        "harmonic_score":    round(plan.harmonic_score, 3) if plan.harmonic_score >= 0 else None,
        # Confidence-gate outcome (DJ_ENGINE_IMPROVEMENT_PLAN.md §2.1) — None
        # when the gate didn't evaluate (stems missing, or harmonic_score
        # unavailable); stem_blend below already reflects the actual path
        # taken post-gate.
        "transition_confidence": (
            round(transition_confidence, 3) if transition_confidence is not None else None
        ),
        "key_a":             struct_a.key_name + " " + struct_a.mode if struct_a.key_name else None,
        "key_b":             struct_b.key_name + " " + struct_b.mode if struct_b.key_name else None,
        "camelot_a":         struct_a.camelot or None,
        "camelot_b":         struct_b.camelot or None,
        "duration_sec":      round(len(mix_mastered) / sr, 1),
        "bridge_beat":       bridge_label,
        "stem_blend":        use_stem_blend,
        "lufs":              round(quality_report.lufs_integrated, 1),
        "peak_dbfs":         round(quality_report.peak_dbfs, 1),
        "dr":                round(quality_report.dynamic_range_db, 1),
        "quality_passed":    quality_report.passed,
        "quality_notes":     quality_report.notes,
        # Spectral cue matching outcome (None when disabled/inapplicable);
        # sims stay server-side in the history store — clients get the summary.
        "spectral_match":    (
            {k: spectral_info.get(k)
             for k in ("score", "spectral", "drop_delta_s", "entry_bar", "original_entry_bar")}
            if spectral_info else None
        ),
        # Dynamic in-transition spectral sync (None when disabled/inapplicable)
        "spectral_sync":     sync_info,
        # Transition timeline in output seconds — what happens when
        "timeline":          _transition_timeline(plan, struct_b),
        # Source-track attribution — also written to out_dir/CREDITS.txt + manifest.json
        "credits":           credits,
        **ts_data,
    }


def task_dj_chain(
    job_id: str,
    songs: list,
    transition_bars: int,
    preset: str,
    bridge_beat_mode: str = "none",
    bridge_beat_genre: str = "auto",
    bridge_beat_intensity: float = 0.38,
    bridge_beat_path: Optional[str] = None,
    transition_effect: str = "auto",
) -> Dict[str, Any]:
    """Render a continuous N-song DJ mix chain."""
    n = len(songs)
    log_audit("dj_chain_start", resource=" → ".join(songs), job_id=job_id,
              metadata={"n_songs": n, "transition_bars": transition_bars,
                        "preset": preset, "effect": transition_effect})
    update_job(job_id, progress=0.02, message=f"Loading {n} songs…")

    eng = _resolve_engine()

    # ── Load audio (full.wav → full_enhanced.wav → stem sum per song) ──────
    # load_source_audio raises FileNotFoundError(name) for any unusable song.
    sr = 44100
    audios = []
    for idx, name in enumerate(songs):
        prog = 0.02 + 0.18 * (idx / n)
        update_job(job_id, progress=prog, message=f"Loading {songs[idx][:30]}…")
        audio, _ = load_source_audio(name, sr=sr, mono=True, duration=180.0)
        audios.append(audio)

    # ── Analyse structure ────────────────────────────────────────────────
    structures = []
    for idx, audio in enumerate(audios):
        prog = 0.20 + 0.25 * (idx / n)
        update_job(job_id, progress=prog, message=f"Analysing structure {idx+1}/{n}…")
        structures.append(eng.analyze(audio, sr))

    # ── Plan transitions ─────────────────────────────────────────────────
    update_job(job_id, progress=0.45, message="Planning transitions…")
    from scripts.core.spectral_matching import apply_spectral_matching
    plans = []
    for i in range(n - 1):
        plan = eng.plan_transition(structures[i], structures[i + 1],
                                   transition_bars=transition_bars)
        # Learned cue matching per adjacent pair (no feedback history for
        # chains — one 👍 can't be attributed to a single transition).
        apply_spectral_matching(plan, audios[i], audios[i + 1], structures[i + 1], sr)
        plans.append(plan)

    # ── Bridge beats (one per transition) ───────────────────────────────
    bridge_audios: list = []
    for i, plan in enumerate(plans):
        bb = None
        if bridge_beat_mode == "auto":
            try:
                from scripts.core.beat_synth import render_beat
                bb = render_beat(
                    bpm=plan.bpm_a,
                    genre=bridge_beat_genre,
                    bars=max(4, transition_bars),
                    sr=sr,
                    intensity=bridge_beat_intensity,
                    build_up=True,
                )
            except Exception:
                bb = None
        elif bridge_beat_mode == "file" and bridge_beat_path:
            try:
                bb, _ = librosa.load(bridge_beat_path, sr=sr, mono=True)
            except Exception:
                bb = None
        bridge_audios.append(bb)

    # ── Render ───────────────────────────────────────────────────────────
    update_job(job_id, progress=0.55, message="Rendering chain mix…")
    engine = eng.DJEngine(sr=sr)
    mix = engine.render_chain(
        tracks=audios,
        plans=plans,
        bridge_beats=bridge_audios,
        bridge_gain=bridge_beat_intensity,
        transition_effect=transition_effect,
    )

    update_job(job_id, progress=0.85, message="Mastering chain mix…")

    # Chain mixes always master to −8 LUFS (club-set loudness) on both
    # engines — only the mastering implementation follows dj.mixing_engine.
    mix_mastered, quality_report = eng.master_mix(mix, sr=sr, target_lufs=-8.0)

    update_job(job_id, progress=0.92, message="Saving output…")

    session_id = str(uuid.uuid4())[:8]
    out_dir    = user_outputs_dir() / f"chain_{session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_label = "_".join(s[:12] for s in songs)[:60]
    out_path   = out_dir / f"chain_{chain_label}.wav"

    sf.write(str(out_path), mix_mastered, sr, subtype="PCM_24")

    credits = _write_mix_credits(out_dir, list(songs))

    # Per-transition summary (include harmonic scores + key data)
    transitions = []
    for i, plan in enumerate(plans):
        tr: dict = {
            "from":             songs[i],
            "to":               songs[i + 1],
            "bpm_from":         round(plan.bpm_a, 1),
            "bpm_to":           round(plan.bpm_b, 1),
            "transition_bars":  plan.transition_bars,
            "transition_sec":   round(plan.transition_seconds, 1),
            "tempo_ratio":      round(plan.tempo_shift_ratio, 4),
            "harmonic_score":   round(plan.harmonic_score, 3) if plan.harmonic_score >= 0 else None,
        }
        if i < len(structures) and structures[i].camelot:
            tr["camelot_from"] = structures[i].camelot
            tr["key_from"] = f"{structures[i].key_name} {structures[i].mode}".strip()
        if (i + 1) < len(structures) and structures[i + 1].camelot:
            tr["camelot_to"] = structures[i + 1].camelot
            tr["key_to"] = f"{structures[i + 1].key_name} {structures[i + 1].mode}".strip()
        transitions.append(tr)

    log_audit("dj_chain_complete", resource=str(out_path), job_id=job_id,
              metadata={"n_songs": n, "duration_sec": round(len(mix_mastered) / sr, 1),
                        "quality_passed": quality_report.passed})
    return {
        "output":          str(out_path),
        "session_id":      session_id,
        # Browser-playable URL — out_dir is "chain_{session_id}", so the path
        # segment must match out_dir.name, not the raw session_id.
        "stream_url":      f"/outputs/{out_dir.name}/{out_path.name}",
        "engine":          eng.name,
        "songs":           songs,
        "n_songs":         n,
        "n_transitions":   n - 1,
        "duration_sec":    round(len(mix_mastered) / sr, 1),
        "bpm_reference":   round(plans[0].bpm_a, 1),
        "bridge_beat":     bridge_beat_mode,
        "transitions":     transitions,
        "lufs":            round(quality_report.lufs_integrated, 1),
        "peak_dbfs":       round(quality_report.peak_dbfs, 1),
        "dr":              round(quality_report.dynamic_range_db, 1),
        "quality_passed":  quality_report.passed,
        "quality_notes":   quality_report.notes,
        # Source-track attribution — also written to out_dir/CREDITS.txt + manifest.json
        "credits":         credits,
    }


# ---------------------------------------------------------------------------
# Remix preview task — transition window only, no full render
# ---------------------------------------------------------------------------

def task_remix_preview(
    job_id: str,
    song_a: str,
    song_b: str,
    transition_bars: int = 16,
    transition_effect: str = "auto",
) -> Dict[str, Any]:
    """
    Render only the transition window between two songs for fast audition.

    Uses DJEngine.render(..., full_output=False) to produce just the crossfade
    segment — typically 16–32 bars (30–60 seconds of audio).  Much faster than
    a full mix render because neither song's full content is processed.

    The preview file is saved to outputs/{session_id}/preview.wav and served
    via GET /outputs/{session_id}/preview.wav.
    """
    log_audit("remix_preview_start", resource=f"{song_a} → {song_b}", job_id=job_id,
              metadata={"transition_bars": transition_bars, "effect": transition_effect})
    update_job(job_id, progress=0.05, message="Loading audio for preview…")

    eng = _resolve_engine()

    update_job(job_id, progress=0.15, message="Analysing tracks…")
    # full.wav → full_enhanced.wav → stem sum; A sets the sample rate, B matches.
    audio_a, sr = load_source_audio(song_a, sr=None, mono=True)
    audio_b, _  = load_source_audio(song_b, sr=sr,  mono=True)

    audio_a = audio_a.astype(np.float32)
    audio_b = audio_b.astype(np.float32)

    update_job(job_id, progress=0.35, message="Planning transition…")
    struct_a = eng.analyze(audio_a, sr)
    struct_b = eng.analyze(audio_b, sr)
    plan = eng.plan_transition(struct_a, struct_b, transition_bars=transition_bars)

    # Same learned cue matching as the full render, so what you audition is
    # what the full remix will do. (Dynamic spectral sync is full-render-only:
    # it profiles the material before/after the overlap, and a preview segment
    # IS the overlap — there's no surrounding context to profile.)
    from scripts.core.spectral_matching import apply_spectral_matching, record_render
    spectral_info = apply_spectral_matching(plan, audio_a, audio_b, struct_b, sr)

    update_job(job_id, progress=0.55, message="Rendering preview segment…")
    engine = eng.DJEngine(sr=sr)
    preview_audio = engine.render(
        audio_a,
        audio_b,
        plan,
        full_output=False,       # ← transition window only
        transition_effect=transition_effect,
    )

    # Normalise to -1 dBFS
    peak = float(np.abs(preview_audio).max())
    if peak > 1e-6:
        preview_audio = preview_audio * (0.891 / peak)  # -1 dBFS

    update_job(job_id, progress=0.85, message="Saving preview…")
    session_id = str(uuid.uuid4())[:8]
    out_dir = user_outputs_dir() / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preview.wav"
    sf.write(str(out_path), preview_audio, sr, subtype="PCM_16")

    if spectral_info:
        record_render(session_id, spectral_info)

    duration_sec = round(len(preview_audio) / sr, 1)
    log_audit("remix_preview_complete", resource=str(out_path), job_id=job_id,
              metadata={"duration_sec": duration_sec, "transition_bars": transition_bars})

    return {
        "output":         str(out_path),
        "session_id":     session_id,
        "filename":       "preview.wav",
        "engine":         eng.name,
        "song_a":         song_a,
        "song_b":         song_b,
        "transition_bars": plan.transition_bars,
        "duration_sec":   duration_sec,
        "bpm_a":          round(struct_a.bpm, 1),
        "bpm_b":          round(struct_b.bpm, 1),
        "harmonic_score": round(plan.harmonic_score, 3) if plan.harmonic_score >= 0 else None,
        # Explainability — tell the user what was detected and why this transition point
        "exit_bar_a":     plan.exit_bar_a,
        "entry_bar_b":    plan.entry_bar_b,
        "camelot_a":      struct_a.camelot if hasattr(struct_a, "camelot") else None,
        "camelot_b":      struct_b.camelot if hasattr(struct_b, "camelot") else None,
        "tempo_ratio":    round(plan.tempo_shift_ratio, 4),
        "spectral_match": (
            {k: spectral_info[k] for k in ("score", "entry_bar", "original_entry_bar")}
            if spectral_info else None
        ),
        "stream_url":     f"/outputs/{session_id}/preview.wav",
    }
