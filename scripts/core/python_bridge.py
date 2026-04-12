#!/usr/bin/env python3
"""
Python-Only Bridge for AI RemixMate

This module provides a complete Python-only remix generation pipeline
without any external DAW dependencies.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


from scripts.core.python_mixer import PythonMixer, MixParameters, MixResult
from scripts.core.musical_analysis import MusicalAnalyzer
from scripts.core.real_optimizer import RealOptimizer
from scripts.core.metrics import AudioMetrics
from scripts.core.features import extract_features_from_wav
from scripts.core.paths import vocals_path, other_path, OUTPUT_DIR, AUDIO_IN, ensure_directories
from scripts.core.youtube_music import fetch_song_to_library, sanitize_name
from scripts.core.genre import auto_preset, detect_genre, get_preset, GenrePreset, GenreResult
from scripts.core.license import load_license, license_warning, LicenseInfo
from scripts.core.library import get_library_manager
from scripts.core.track_metadata import MetadataClient, TrackMetadata, quick_lookup
from scripts.core.dj_engine import (
    DJEngine, SongStructure, TransitionPlan,
    _analyze_impl as _dj_analyze, plan_transition as _dj_plan,
)
from scripts.core.config import cfg


class PythonBridge:
    """Python-only bridge for remix generation."""
    
    def __init__(self, sample_rate: int | None = None, bit_depth: int | None = None):
        self.sr         = sample_rate or cfg.audio.sample_rate
        self.bit_depth  = bit_depth   or cfg.audio.bit_depth

        # Initialize components
        self.mixer            = PythonMixer(self.sr, self.bit_depth)
        self.musical_analyzer = MusicalAnalyzer(self.sr)
        self.optimizer        = RealOptimizer(self.sr)
        self.metrics          = AudioMetrics()
        self.metadata_client  = MetadataClient()
        self.dj_engine        = DJEngine(self.sr)
    
    def create_remix(self, base_song: str, match_song: str,
                    output_dir: str, preset: str = "auto",
                    optimize: bool = False, online: bool = True) -> Dict[str, Any]:
        """
        Create a complete remix using only Python.
        
        Args:
            base_song: Base song name (vocals source)
            match_song: Match song name (instrumentals source)
            output_dir: Output directory for results
            preset: Mix preset — "auto" (detect from audio), or a genre name
                    like "house", "hiphop", "pop", "trap", "techno", "rnb",
                    "dnb", "ambient", "rock", "jazz".
                    Legacy names "radio"→pop, "club"→house still work.
            optimize: Whether to use iterative optimization

        Returns:
            Dictionary with remix results, quality metrics, genre info, and license warnings.
        """
        print(f"🎵 Creating Python-only remix: {base_song} + {match_song}")
        print(f"   Preset: {preset}, Optimization: {optimize}")
        
        start_time = time.time()
        
        try:
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get file paths
            vocals_file = vocals_path(base_song)
            instrumentals_file = other_path(match_song)
            
            # Fallbacks when stems are missing: use AUDIO_IN WAV if present
            if not vocals_file.exists():
                alt_vocals = AUDIO_IN / f"{base_song}.wav"
                if alt_vocals.exists():
                    vocals_file = alt_vocals
            if not instrumentals_file.exists():
                alt_inst = AUDIO_IN / f"{match_song}.wav"
                if alt_inst.exists():
                    instrumentals_file = alt_inst
            
            # If still missing, try fetching via YouTube Music
            missing = []
            if not vocals_file.exists():
                missing.append(("vocals", base_song))
            if not instrumentals_file.exists():
                missing.append(("instrumentals", match_song))
            
            if missing and online:
                print("🌐 Missing local files; attempting YouTube Music fetch...")
                for kind, name in missing:
                    res = fetch_song_to_library(name, out_name=name, separate=True)
                    if kind == "vocals":
                        # prefer stems if available else wav
                        if "vocals" in res:
                            vocals_file = res["vocals"]
                        elif "other" in res:
                            vocals_file = res["other"]
                        elif "wav" in res:
                            vocals_file = res["wav"]
                    else:
                        if "other" in res:
                            instrumentals_file = res["other"]
                        elif "wav" in res:
                            instrumentals_file = res["wav"]
            
            if not vocals_file.exists() or not instrumentals_file.exists():
                raise FileNotFoundError("Required audio files not found (local or fetched)")
            
            # Load and analyze audio
            print("📊 Loading and analyzing audio...")
            vocals, _ = self.mixer.load_audio(vocals_file)
            instrumentals, _ = self.mixer.load_audio(instrumentals_file)

            # ── Genre detection / preset selection ─────────────────────────
            genre_result: Optional[GenreResult] = None
            if preset == "auto":
                print("🎸 Auto-detecting genre from audio...")
                genre_result = detect_genre(vocals, self.sr)
                genre_preset = genre_result.preset
                print(
                    f"   Detected: {genre_preset.display_name} "
                    f"({genre_result.confidence:.0%} confidence)"
                )
                if genre_result.runner_up:
                    print(f"   Runner-up: {genre_result.runner_up}")
            else:
                genre_preset = get_preset(preset)
                print(f"🎸 Preset: {genre_preset.display_name} ({genre_preset.genre})")

            # ── License warnings ────────────────────────────────────────────
            for song_name, song_file in (
                (base_song, vocals_file), (match_song, instrumentals_file)
            ):
                try:
                    lic_info = load_license(song_file.parent)
                    if lic_info:
                        warn = license_warning(lic_info, song_name)
                        if warn:
                            print(warn)
                except Exception:
                    pass  # licence check is best-effort; never block a remix

            # ── Library touch (mark as accessed) ───────────────────────────
            try:
                mgr = get_library_manager()
                mgr.touch(base_song)
                mgr.touch(match_song)
            except Exception:
                pass

            # Apply musical corrections
            print("🎼 Applying musical corrections...")
            vocals_corrected, instrumentals_corrected, correction_info = self.musical_analyzer.apply_musical_corrections(
                vocals, instrumentals
            )

            # Build mix parameters from genre preset
            mix_params = self._genre_to_mix_params(genre_preset)
            
            # Optimize parameters if requested
            if optimize:
                print("🔄 Optimizing mix parameters...")
                optimization_result = self.optimizer.optimize(
                    vocals_corrected, instrumentals_corrected, max_iterations=50
                )
                
                # Update mix parameters with optimized values
                mix_params.vocal_gain_db = optimization_result.best_parameters.vocal_gain
                mix_params.instrumental_gain_db = optimization_result.best_parameters.instrumental_gain
                mix_params.sidechain_amount = optimization_result.best_parameters.sidechain_amount
                mix_params.hp_filter_freq = optimization_result.best_parameters.eq_low_cut_freq
                mix_params.reverb_send = optimization_result.best_parameters.reverb_send
            
            # Mix the audio
            print("🎛️ Mixing audio...")
            mix_result = self.mixer.mix_audio(vocals_corrected, instrumentals_corrected, mix_params)
            
            if not mix_result.success:
                raise RuntimeError(f"Mixing failed: {mix_result.error_message}")
            
            # Save the mix
            mix_output_path = output_path / f"{base_song}_x_{match_song}_{preset}.wav"
            if not self.mixer.save_mix(mix_result, mix_output_path):
                raise RuntimeError("Failed to save mix")
            
            # Calculate comprehensive metrics
            print("📊 Calculating quality metrics...")
            quality_metrics = self.metrics.evaluate_remix(
                vocals_file, instrumentals_file, mix_output_path
            )
            
            # Create session report
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            report = {
                "session_id": session_id,
                "base_song": base_song,
                "match_song": match_song,
                "preset": preset,
                "optimization_used": optimize,
                "processing_time": time.time() - start_time,
                "genre": {
                    "genre": genre_preset.genre,
                    "display_name": genre_preset.display_name,
                    "auto_detected": preset == "auto",
                    "confidence": genre_result.confidence if genre_result else 1.0,
                    "runner_up": genre_result.runner_up if genre_result else None,
                },
                "mix_parameters": {
                    "vocal_gain_db": mix_params.vocal_gain_db,
                    "instrumental_gain_db": mix_params.instrumental_gain_db,
                    "sidechain_amount": mix_params.sidechain_amount,
                    "hp_filter_freq": mix_params.hp_filter_freq,
                    "reverb_send": mix_params.reverb_send,
                    "master_limiter_threshold": mix_params.master_limiter_threshold
                },
                "musical_corrections": correction_info,
                "quality_metrics": quality_metrics,
                "mix_result": {
                    "processing_time": mix_result.processing_time,
                    "quality_metrics": mix_result.quality_metrics
                },
                "files": {
                    "vocals_source": str(vocals_file),
                    "instrumentals_source": str(instrumentals_file),
                    "mix_output": str(mix_output_path)
                },
                "success": True
            }
            
            # Save report
            report_path = output_path / f"remix_report_{session_id}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✅ Remix completed successfully!")
            print(f"   Output: {mix_output_path}")
            print(f"   Report: {report_path}")
            print(f"   Processing time: {report['processing_time']:.2f}s")
            
            return report
            
        except Exception as e:
            error_report = {
                "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "base_song": base_song,
                "match_song": match_song,
                "preset": preset,
                "optimization_used": optimize,
                "processing_time": time.time() - start_time,
                "success": False,
                "error": str(e)
            }
            
            # Save error report
            error_path = Path(output_dir) / f"error_report_{error_report['session_id']}.json"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_path, 'w') as f:
                json.dump(error_report, f, indent=2)
            
            print(f"❌ Remix failed: {e}")
            return error_report
    
    def _genre_to_mix_params(self, genre_preset: GenrePreset) -> MixParameters:
        """
        Convert a GenrePreset into MixParameters for the audio mixer.

        GenrePreset carries the full genre-specific configuration; we map the
        relevant fields straight across and use sensible fixed defaults for
        the few parameters the preset doesn't expose (e.g. compression_threshold).
        """
        return MixParameters(
            vocal_gain_db=genre_preset.vocal_gain_db,
            instrumental_gain_db=genre_preset.inst_gain_db,
            sidechain_amount=genre_preset.inst_sidechain_amount,
            hp_filter_freq=genre_preset.vocal_hp_filter_hz,
            reverb_send=genre_preset.vocal_reverb_send,
            master_limiter_threshold=genre_preset.true_peak_ceiling,
            compression_ratio=genre_preset.inst_compression_ratio,
            compression_threshold=-12.0,  # not genre-specific; left at default
        )

    def _get_preset_parameters(self, preset: str) -> MixParameters:
        """
        [Deprecated] Get mix parameters for a legacy preset name.

        Use _genre_to_mix_params(get_preset(name)) instead.
        Kept for backward compatibility with any external callers.
        """
        return self._genre_to_mix_params(get_preset(preset))

    # ------------------------------------------------------------------
    # Metadata-first API
    # ------------------------------------------------------------------

    def check_compatibility(
        self,
        song_a: str,
        song_b: str,
        artist_a: str = "",
        artist_b: str = "",
    ) -> Dict[str, Any]:
        """
        Check mixing compatibility between two songs using API metadata only.
        No audio download required — runs in under 3 seconds.

        Returns a compatibility report with key/BPM/energy scores and a
        human-readable verdict.
        """
        meta_a = self.metadata_client.lookup(song_a, artist_a)
        meta_b = self.metadata_client.lookup(song_b, artist_b)
        compat = self.metadata_client.compatibility_score(meta_a, meta_b)

        verdict = (
            "✅ Great mix" if compat["overall"] >= 0.75 else
            "🟡 Workable mix" if compat["overall"] >= 0.55 else
            "🔴 Difficult mix — consider a different pairing"
        )

        return {
            "song_a": {"title": meta_a.title, "artist": meta_a.artist,
                       "bpm": meta_a.bpm, "key": meta_a.key_full,
                       "camelot": meta_a.camelot, "genres": meta_a.genres},
            "song_b": {"title": meta_b.title, "artist": meta_b.artist,
                       "bpm": meta_b.bpm, "key": meta_b.key_full,
                       "camelot": meta_b.camelot, "genres": meta_b.genres},
            "compatibility": compat,
            "verdict": verdict,
            "meta_a": meta_a,
            "meta_b": meta_b,
        }

    def dj_remix(
        self,
        base_song: str,
        match_song: str,
        output_dir: str,
        transition_bars: int = 16,
        preset: str = "auto",
    ) -> Dict[str, Any]:
        """
        Create a full DJ-style transition mix.

        Flow:
          1. Metadata lookup (instant, API)   — BPM, key, genre
          2. Compatibility check              — key distance, BPM ratio
          3. Load audio + structure analysis  — phrase/section detection
          4. Transition planning              — phrase-aligned mix point, EQ plan
          5. Render                           — HP ramp, bass swap, crossfade
          6. Post-process                     — apply genre preset, master

        Parameters
        ----------
        base_song : str
            Song providing the VOCALS (outgoing track).
        match_song : str
            Song providing the INSTRUMENTALS (incoming track).
        output_dir : str
            Where to write the output mix.
        transition_bars : int
            Length of the DJ transition overlap (8 / 16 / 32 bars).
        preset : str
            Genre preset or "auto" for auto-detection.
        """
        start_time = time.time()
        ensure_directories()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # ── Step 1: Metadata lookup ────────────────────────────────
            print("🔍 Fetching metadata…")
            meta_a = self.metadata_client.lookup(base_song)
            meta_b = self.metadata_client.lookup(match_song)
            compat = self.metadata_client.compatibility_score(meta_a, meta_b)
            print(f"   {base_song}: {meta_a.bpm:.1f} BPM, {meta_a.key_full}, "
                  f"Camelot {meta_a.camelot or '?'}")
            print(f"   {match_song}: {meta_b.bpm:.1f} BPM, {meta_b.key_full}, "
                  f"Camelot {meta_b.camelot or '?'}")
            print(f"   Compatibility: {compat['overall']:.0%} "
                  f"(key={compat['key_score']:.0%}, "
                  f"bpm={compat['bpm_score']:.0%})")

            # ── Step 2: Load stems ─────────────────────────────────────
            print("🎵 Loading audio stems…")
            vocals_file = vocals_path(base_song)
            inst_file   = other_path(match_song)

            if not vocals_file.exists():
                raise FileNotFoundError(
                    f"Vocals not found for '{base_song}'. "
                    f"Run download with --split first."
                )
            if not inst_file.exists():
                raise FileNotFoundError(
                    f"Instrumentals not found for '{match_song}'. "
                    f"Run download with --split first."
                )

            vocals, _   = self.mixer.load_audio(str(vocals_file))
            inst, _     = self.mixer.load_audio(str(inst_file))

            # Enrich metadata with local analysis (fills any gaps)
            meta_a = self.metadata_client.enrich_from_file(meta_a, vocals_file)
            meta_b = self.metadata_client.enrich_from_file(meta_b, inst_file)

            # ── Step 3: License warnings ───────────────────────────────
            for song_name, wav_file in ((base_song, vocals_file), (match_song, inst_file)):
                try:
                    lic = load_license(wav_file.parent)
                    if lic:
                        warn = license_warning(lic, song_name)
                        if warn:
                            print(warn)
                except Exception:
                    pass

            # ── Step 4: Structure analysis ─────────────────────────────
            print("🏗️  Analysing musical structure…")
            structure_a = _dj_analyze(vocals, self.sr)
            structure_b = _dj_analyze(inst, self.sr)
            print(f"   {base_song}: {structure_a.total_bars} bars, "
                  f"sections: {' → '.join(s.type for s in structure_a.sections)}")
            print(f"   {match_song}: {structure_b.total_bars} bars, "
                  f"sections: {' → '.join(s.type for s in structure_b.sections)}")

            # ── Step 5: Transition planning ────────────────────────────
            print("🎛️  Planning DJ transition…")
            plan = _dj_plan(structure_a, structure_b,
                            transition_bars=transition_bars)
            print(f"   Exit at bar {plan.exit_bar_a} ({plan.exit_time_a:.1f}s)")
            print(f"   {plan.transition_bars}-bar overlap, "
                  f"bass swap at bar {plan.eq.bass_swap_bar}")

            # ── Step 6: Genre preset ───────────────────────────────────
            genre_result: Optional[GenreResult] = None
            if preset == "auto":
                genre_result = detect_genre(vocals, self.sr)
                genre_preset = genre_result.preset
                print(f"🎸 Genre: {genre_preset.display_name} "
                      f"({genre_result.confidence:.0%})")
            else:
                genre_preset = get_preset(preset)
                print(f"🎸 Preset: {genre_preset.display_name}")

            # ── Step 7: Render DJ transition ───────────────────────────
            print("🎚️  Rendering transition…")
            mixed = self.dj_engine.render(vocals, inst, plan)

            # ── Step 8: Master with genre preset ──────────────────────
            mix_params  = self._genre_to_mix_params(genre_preset)
            mix_result  = self.mixer.mix_audio(mixed, mixed, mix_params)
            final_audio = mix_result.mixed_audio if mix_result.success else mixed

            # ── Step 9: Write output ───────────────────────────────────
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name   = (f"dj_{session_id}_"
                          f"{base_song[:20].replace(' ', '_')}_into_"
                          f"{match_song[:20].replace(' ', '_')}.wav")
            out_file   = output_path / out_name

            import soundfile as sf
            sf.write(str(out_file), final_audio, self.sr, subtype="PCM_24")
            print(f"✅ DJ mix saved: {out_file.name}")

            # ── Library housekeeping ───────────────────────────────────
            try:
                mgr = get_library_manager()
                mgr.touch(base_song)
                mgr.touch(match_song)
            except Exception:
                pass

            return {
                "session_id": session_id,
                "type": "dj_remix",
                "base_song": base_song,
                "match_song": match_song,
                "processing_time": time.time() - start_time,
                "metadata": {
                    "song_a": {"bpm": meta_a.bpm, "key": meta_a.key_full,
                               "camelot": meta_a.camelot},
                    "song_b": {"bpm": meta_b.bpm, "key": meta_b.key_full,
                               "camelot": meta_b.camelot},
                },
                "compatibility": compat,
                "transition": {
                    "exit_bar": plan.exit_bar_a,
                    "exit_time": plan.exit_time_a,
                    "transition_bars": plan.transition_bars,
                    "bass_swap_bar": plan.eq.bass_swap_bar,
                },
                "genre": {
                    "genre": genre_preset.genre,
                    "display_name": genre_preset.display_name,
                    "auto_detected": preset == "auto",
                    "confidence": genre_result.confidence if genre_result else 1.0,
                },
                "files": {"mix_output": str(out_file)},
                "success": True,
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "processing_time": time.time() - start_time,
            }

    def batch_remix(self, remix_pairs: list, output_dir: str,
                   preset: str = "radio", optimize: bool = False) -> Dict[str, Any]:
        """
        Create multiple remixes in batch.
        
        Args:
            remix_pairs: List of (base_song, match_song) tuples
            output_dir: Output directory for results
            preset: Mix preset
            optimize: Whether to use optimization
            
        Returns:
            Dictionary with batch results
        """
        print(f"🎵 Starting batch remix: {len(remix_pairs)} pairs")
        
        results = []
        successful = 0
        failed = 0
        
        for i, (base_song, match_song) in enumerate(remix_pairs, 1):
            print(f"\n📊 Processing {i}/{len(remix_pairs)}: {base_song} + {match_song}")
            
            try:
                result = self.create_remix(base_song, match_song, output_dir, preset, optimize)
                results.append(result)
                
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"❌ Failed: {e}")
                failed += 1
                results.append({
                    "base_song": base_song,
                    "match_song": match_song,
                    "success": False,
                    "error": str(e)
                })
        
        # Create batch summary
        batch_summary = {
            "total_pairs": len(remix_pairs),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(remix_pairs) if remix_pairs else 0,
            "preset": preset,
            "optimization_used": optimize,
            "results": results
        }
        
        # Save batch report
        batch_report_path = Path(output_dir) / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_report_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"\n✅ Batch remix completed!")
        print(f"   Successful: {successful}/{len(remix_pairs)}")
        print(f"   Failed: {failed}")
        print(f"   Success rate: {batch_summary['success_rate']:.1%}")
        print(f"   Batch report: {batch_report_path}")
        
        return batch_summary


def main():
    """Command-line interface for Python bridge."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Python-Only AI RemixMate Bridge")
    parser.add_argument("--base", required=True, help="Base song name (vocals source)")
    parser.add_argument("--match", required=True, help="Match song name (instrumentals source)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--preset", choices=["radio", "club", "ambient"], default="radio", help="Mix preset")
    parser.add_argument("--optimize", action="store_true", help="Use iterative optimization")
    parser.add_argument("--batch", help="Batch file with remix pairs (JSON)")
    
    args = parser.parse_args()
    
    # Create bridge
    bridge = PythonBridge()
    
    if args.batch:
        # Batch processing
        with open(args.batch) as f:
            batch_data = json.load(f)
        remix_pairs = batch_data.get("remix_pairs", [])
        result = bridge.batch_remix(remix_pairs, args.output_dir, args.preset, args.optimize)
    else:
        # Single remix
        result = bridge.create_remix(args.base, args.match, args.output_dir, args.preset, args.optimize)
    
    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    exit(main())
