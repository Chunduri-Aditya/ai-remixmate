"""
scripts/api/task_modules/remix.py — DJ remix and chain task functions.

task_dj_remix          — render a 2-song DJ transition mix (full output)
task_dj_chain          — render an N-song continuous DJ mix
task_remix_preview     — render only the transition window for fast audition
"""

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np
import soundfile as sf

from scripts.api.jobs import update_job
from scripts.core.audit import log_audit
from scripts.core.dj_engine import _analyze_impl, plan_transition, DJEngine
from scripts.core.genre import auto_preset
from scripts.core.paths import OUTPUTS_DIR, song_dir


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

    wav_a = song_dir(song_a) / "full.wav"
    wav_b = song_dir(song_b) / "full.wav"
    if not wav_a.exists():
        raise FileNotFoundError(f"Song not in library: {song_a}")
    if not wav_b.exists():
        raise FileNotFoundError(f"Song not in library: {song_b}")

    sr = 44100
    audio_a, _ = librosa.load(str(wav_a), sr=sr, mono=True, duration=180.0)
    audio_b, _ = librosa.load(str(wav_b), sr=sr, mono=True, duration=180.0)

    update_job(job_id, progress=0.25, message="Analysing structure…")

    struct_a = _analyze_impl(audio_a, sr)
    struct_b = _analyze_impl(audio_b, sr)

    update_job(job_id, progress=0.45, message="Planning transition…")

    plan = plan_transition(struct_a, struct_b, transition_bars=transition_bars)

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
    use_stem_blend = has_stems_a and has_stems_b

    if use_stem_blend:
        update_job(job_id, progress=0.55,
                   message="Rendering stem-aware DJ mix (intelligent instrument blending)…")
    else:
        update_job(job_id, progress=0.55, message="Rendering DJ mix…")

    engine = DJEngine(sr=sr)

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

    update_job(job_id, progress=0.80, message="Mastering output…")

    from scripts.core.mastering import master_mix
    mix_mastered, quality_report = master_mix(mix, sr=sr, target_lufs=-8.0)   # DJ mix target

    update_job(job_id, progress=0.90, message="Saving output…")

    session_id = str(uuid.uuid4())[:8]
    out_dir = OUTPUTS_DIR / f"dj_{session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dj_{song_a[:20]}_{song_b[:20]}.wav"

    sf.write(str(out_path), mix_mastered, sr, subtype="PCM_24")

    # Compute AI transition score if music intelligence features are available
    ts_data: dict = {}
    try:
        from scripts.core.music_intelligence import compute_track_vector, compute_transition_score
        vec_a = compute_track_vector(audio_a, sr, bpm=struct_a.bpm)
        vec_b = compute_track_vector(audio_b, sr, bpm=struct_b.bpm)
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
    except Exception as exc:
        pass   # non-critical

    log_audit("dj_remix_complete", resource=str(out_path), job_id=job_id,
              metadata={"song_a": song_a, "song_b": song_b, "stem_blend": use_stem_blend})
    return {
        "output":            str(out_path),
        "session_id":        session_id,
        "song_a":            song_a,
        "song_b":            song_b,
        "bpm_a":             round(plan.bpm_a, 1),
        "bpm_b":             round(plan.bpm_b, 1),
        "transition_bars":   plan.transition_bars,
        "transition_sec":    round(plan.transition_seconds, 1),
        "tempo_ratio":       round(plan.tempo_shift_ratio, 4),
        "harmonic_score":    round(plan.harmonic_score, 3) if plan.harmonic_score >= 0 else None,
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

    # ── Validate all songs exist ──────────────────────────────────────────
    wavs = []
    for name in songs:
        wav = song_dir(name) / "full.wav"
        if not wav.exists():
            raise FileNotFoundError(f"Song not in library: {name}")
        wavs.append(wav)

    # ── Load audio ───────────────────────────────────────────────────────
    sr = 44100
    audios = []
    for idx, wav in enumerate(wavs):
        prog = 0.02 + 0.18 * (idx / n)
        update_job(job_id, progress=prog, message=f"Loading {songs[idx][:30]}…")
        audio, _ = librosa.load(str(wav), sr=sr, mono=True, duration=180.0)
        audios.append(audio)

    # ── Analyse structure ────────────────────────────────────────────────
    structures = []
    for idx, audio in enumerate(audios):
        prog = 0.20 + 0.25 * (idx / n)
        update_job(job_id, progress=prog, message=f"Analysing structure {idx+1}/{n}…")
        structures.append(_analyze_impl(audio, sr))

    # ── Plan transitions ─────────────────────────────────────────────────
    update_job(job_id, progress=0.45, message="Planning transitions…")
    plans = []
    for i in range(n - 1):
        plans.append(plan_transition(structures[i], structures[i + 1],
                                     transition_bars=transition_bars))

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
    engine = DJEngine(sr=sr)
    mix = engine.render_chain(
        tracks=audios,
        plans=plans,
        bridge_beats=bridge_audios,
        bridge_gain=bridge_beat_intensity,
        transition_effect=transition_effect,
    )

    update_job(job_id, progress=0.85, message="Mastering chain mix…")

    from scripts.core.mastering import master_mix as _master_mix
    mix_mastered, quality_report = _master_mix(mix, sr=sr, target_lufs=-8.0)

    update_job(job_id, progress=0.92, message="Saving output…")

    session_id = str(uuid.uuid4())[:8]
    out_dir    = OUTPUTS_DIR / f"chain_{session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    chain_label = "_".join(s[:12] for s in songs)[:60]
    out_path   = out_dir / f"chain_{chain_label}.wav"

    sf.write(str(out_path), mix_mastered, sr, subtype="PCM_24")

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

    wav_a = song_dir(song_a) / "full.wav"
    wav_b = song_dir(song_b) / "full.wav"

    for label, path in ((song_a, wav_a), (song_b, wav_b)):
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found for '{label}': {path}")

    update_job(job_id, progress=0.15, message="Analysing tracks…")
    audio_a, sr = librosa.load(str(wav_a), sr=None, mono=True)
    audio_b, _  = librosa.load(str(wav_b), sr=sr,   mono=True)

    audio_a = audio_a.astype(np.float32)
    audio_b = audio_b.astype(np.float32)

    update_job(job_id, progress=0.35, message="Planning transition…")
    struct_a = _analyze_impl(audio_a, sr)
    struct_b = _analyze_impl(audio_b, sr)
    plan = plan_transition(struct_a, struct_b, transition_bars=transition_bars)

    update_job(job_id, progress=0.55, message="Rendering preview segment…")
    engine = DJEngine(sr=sr)
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
    out_dir = OUTPUTS_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preview.wav"
    sf.write(str(out_path), preview_audio, sr, subtype="PCM_16")

    duration_sec = round(len(preview_audio) / sr, 1)
    log_audit("remix_preview_complete", resource=str(out_path), job_id=job_id,
              metadata={"duration_sec": duration_sec, "transition_bars": transition_bars})

    return {
        "output":         str(out_path),
        "session_id":     session_id,
        "filename":       "preview.wav",
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
        "stream_url":     f"/outputs/{session_id}/preview.wav",
    }
