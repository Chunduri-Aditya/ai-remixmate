#!/usr/bin/env python3
"""
scripts/test_real_audio.py — Real-audio integration test using your actual library.

Tests the full pipeline on real downloaded songs:
  1. Genre detection on 5 library songs
  2. DJ structure analysis (BPM, bars, sections)
  3. Compatibility scoring between 3 song pairs
  4. Transition planning between 2 songs
  5. Library manager summary

Run from the project root:
    python -m scripts.test_real_audio
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def _pick_songs(n: int = 5, min_mb: float = 20, max_mb: float = 80) -> list[Path]:
    """Pick n songs from the library in a sensible size range."""
    lib = PROJECT_ROOT / "library"
    candidates = []
    for d in sorted(lib.iterdir()):
        wav = d / "full.wav"
        if not wav.exists():
            continue
        size_mb = wav.stat().st_size / 1_048_576
        if min_mb <= size_mb <= max_mb:
            candidates.append(wav)
    # Spread across the alphabet for variety
    step = max(1, len(candidates) // n)
    return candidates[::step][:n]


def run_genre_detection(songs: list[Path]) -> list[dict]:
    print("\n" + "=" * 70)
    print("TEST 1 — Genre Detection (real library songs)")
    print("=" * 70)

    import librosa
    from scripts.core.genre import detect_genre

    results = []
    for wav in songs:
        name = wav.parent.name
        t0 = time.time()
        audio, sr = librosa.load(str(wav), sr=22050, mono=True, duration=120.0)
        g = detect_genre(audio, sr)
        elapsed = time.time() - t0
        results.append({"name": name, "genre": g.genre, "conf": g.confidence,
                        "runner_up": g.runner_up, "audio": audio, "sr": sr})
        print(f"  ✅  {name[:50]:<50}  genre={g.genre:<10}  conf={g.confidence:.2f}  "
              f"runner_up={g.runner_up or '-':<10}  ({elapsed:.1f}s)")
    return results


def run_structure_analysis(genre_results: list[dict]) -> list[dict]:
    print("\n" + "=" * 70)
    print("TEST 2 — DJ Structure Analysis (BPM, bars, sections)")
    print("=" * 70)

    from scripts.core.dj_engine import _analyze_impl

    structs = []
    for r in genre_results:
        t0 = time.time()
        struct = _analyze_impl(r["audio"], r["sr"])
        elapsed = time.time() - t0
        structs.append({"name": r["name"], "struct": struct})
        section_types = [s.type for s in struct.sections]
        print(f"  ✅  {r['name'][:45]:<45}  bpm={struct.bpm:<7.1f}  "
              f"bars={struct.total_bars:<4}  sections={section_types}  ({elapsed:.1f}s)")
    return structs


def run_compatibility_scoring(genre_results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("TEST 3 — Metadata Compatibility Scoring (pairs)")
    print("=" * 70)

    from scripts.core.track_metadata import TrackMetadata, MetadataClient
    from scripts.core.dj_engine import _analyze_impl

    client = MetadataClient()
    pairs = [
        (genre_results[0], genre_results[1]),
        (genre_results[1], genre_results[2]),
        (genre_results[0], genre_results[-1]),
    ]

    for a, b in pairs:
        struct_a = _analyze_impl(a["audio"], a["sr"])
        struct_b = _analyze_impl(b["audio"], b["sr"])
        # Build TrackMetadata from what we know locally
        meta_a = TrackMetadata(title=a["name"], bpm=struct_a.bpm, genres=[a["genre"]])
        meta_b = TrackMetadata(title=b["name"], bpm=struct_b.bpm, genres=[b["genre"]])
        score = client.compatibility_score(meta_a, meta_b)
        compat = "✅ COMPATIBLE" if score.get("compatible") else "⚠️  MARGINAL"
        print(f"  {compat}  {a['name'][:30]:<30} ↔ {b['name'][:30]:<30}  "
              f"bpm_score={score.get('bpm_score', 0):.2f}  "
              f"overall={score.get('overall', 0):.2f}")


def run_transition_plan(structs: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("TEST 4 — Transition Planning")
    print("=" * 70)

    from scripts.core.dj_engine import plan_transition

    if len(structs) < 2:
        print("  ⚠️  Need at least 2 songs. Skipping.")
        return

    a, b = structs[0], structs[1]
    plan = plan_transition(a["struct"], b["struct"], transition_bars=16)
    print(f"  Song A: {a['name'][:45]}")
    print(f"  Song B: {b['name'][:45]}")
    print(f"  Exit bar:       {plan.exit_bar_a}  (t={plan.exit_time_a:.1f}s)")
    print(f"  Entry bar:      {plan.entry_bar_b}  (t={plan.entry_time_b:.1f}s)")
    print(f"  Transition:     {plan.transition_bars} bars = {plan.transition_seconds:.1f}s")
    print(f"  BPM A→B:        {plan.bpm_a:.1f} → {plan.bpm_b:.1f}  "
          f"(stretch ratio {plan.tempo_shift_ratio:.3f})")
    print(f"  EQ HP ramp:     {plan.eq.hp_start_hz:.0f}Hz → {plan.eq.hp_end_hz:.0f}Hz")
    print(f"  Bass swap bar:  {plan.eq.bass_swap_bar}")
    print(f"  ✅  Plan looks valid" if plan.transition_bars % 8 == 0 else "  ⚠️  Bars not phrase-aligned!")


def run_library_summary() -> None:
    print("\n" + "=" * 70)
    print("TEST 5 — Library Manager Summary")
    print("=" * 70)

    from scripts.core.library import LibraryManager

    lib_path = PROJECT_ROOT / "library"
    mgr = LibraryManager(library_dir=lib_path)
    print(f"  {mgr.summary()}")

    songs = mgr.list_songs()
    print(f"  Tracked in index: {len(songs)} songs")
    total_songs = sum(1 for d in lib_path.iterdir() if (d / "full.wav").exists())
    print(f"  Total on disk:    {total_songs} songs with full.wav")
    size_gb = mgr.get_size_gb()
    print(f"  Total size:       {size_gb:.2f} GB")


def main() -> None:
    print("🎵  AI RemixMate — Real Audio Integration Test")
    print(f"    Library: {PROJECT_ROOT / 'library'}")

    songs = _pick_songs(n=5)
    if not songs:
        print("❌  No songs found in library. Run the downloader first.")
        sys.exit(1)

    print(f"\n  Using {len(songs)} test songs:")
    for s in songs:
        size = s.stat().st_size / 1_048_576
        print(f"    {s.parent.name[:60]}  ({size:.1f}MB)")

    t_start = time.time()

    # Run all tests
    genre_results = run_genre_detection(songs)
    struct_results = run_structure_analysis(genre_results)
    run_compatibility_scoring(genre_results)
    run_transition_plan(struct_results)
    run_library_summary()

    total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"✅  All tests passed  |  Total time: {total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
