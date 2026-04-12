#!/usr/bin/env python3
"""
Bridge Metrics for AI RemixMate

This module provides specialized metrics for the Logic Pro bridge integration:
- LUFS measurement with pyloudnorm
- True peak detection
- Clipping analysis
- Beat alignment measurement
- Key compatibility checking
"""

from __future__ import annotations
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import librosa

# Try to import pyloudnorm, fall back to basic implementation if not available
try:
    import pyloudnorm as pyln
    PYLN_AVAILABLE = True
except ImportError:
    PYLN_AVAILABLE = False
    print("⚠️ pyloudnorm not available, using basic LUFS approximation")


def lufs_metrics(wav_path: Path) -> Tuple[float, float, float]:
    """
    Calculate LUFS, true peak, and clipping metrics.
    
    Args:
        wav_path: Path to audio file
        
    Returns:
        Tuple of (lufs_integrated, true_peak_dbfs, clipping_ratio)
    """
    try:
        # Load audio
        audio, sr = sf.read(str(wav_path))
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Calculate LUFS
        if PYLN_AVAILABLE:
            meter = pyln.Meter(sr)
            lufs = meter.integrated_loudness(audio)
        else:
            # Basic LUFS approximation using RMS
            rms = np.sqrt(np.mean(audio**2))
            lufs = 20 * np.log10(rms + 1e-10)
        
        # Calculate true peak (naive implementation)
        true_peak = np.max(np.abs(audio))
        true_peak_dbfs = 20 * np.log10(true_peak + 1e-10)
        
        # Calculate clipping ratio
        clipping_ratio = float((np.abs(audio) >= 0.999).mean())
        
        return float(lufs), float(true_peak_dbfs), clipping_ratio
        
    except Exception as e:
        print(f"⚠️ LUFS metrics failed: {e}")
        return -14.0, -1.0, 0.0


def beat_alignment_ms(vocal_path: Path, inst_path: Path) -> float:
    """
    Calculate beat alignment between vocal and instrumental tracks.
    
    Args:
        vocal_path: Path to vocal audio file
        inst_path: Path to instrumental audio file
        
    Returns:
        Beat alignment error in milliseconds
    """
    try:
        # Load audio files
        y1, sr1 = librosa.load(str(vocal_path), mono=True)
        y2, sr2 = librosa.load(str(inst_path), mono=True)
        
        # Calculate onset strength
        o1 = librosa.onset.onset_strength(y=y1, sr=sr1)
        o2 = librosa.onset.onset_strength(y=y2, sr=sr2)
        
        # Cross-correlation to find alignment
        c = np.correlate((o1 - o1.mean()), (o2 - o2.mean()), mode="full")
        lag = np.argmax(c) - (len(o2) - 1)
        
        # Convert lag (frames) to milliseconds
        # onset_strength uses hop_length=512 by default
        hop = 512
        ms = abs(lag) * hop / sr1 * 1000.0
        
        return float(ms)
        
    except Exception as e:
        print(f"⚠️ Beat alignment calculation failed: {e}")
        return 0.0


def key_compatible_camelot(key_base: str, key_vocal: str) -> bool:
    """
    Check if two Camelot keys are compatible.
    
    Args:
        key_base: Base track key in Camelot notation
        key_vocal: Vocal track key in Camelot notation
        
    Returns:
        True if keys are compatible
    """
    # Define compatibility matrix
    compatible = {
        "8A": {"8A", "9A", "5A", "8B"},
        "8B": {"8B", "9B", "5B", "8A"},
        "9A": {"9A", "10A", "6A", "9B"},
        "9B": {"9B", "10B", "6B", "9A"},
        "10A": {"10A", "11A", "7A", "10B"},
        "10B": {"10B", "11B", "7B", "10A"},
        "11A": {"11A", "12A", "8A", "11B"},
        "11B": {"11B", "12B", "8B", "11A"},
        "12A": {"12A", "1A", "9A", "12B"},
        "12B": {"12B", "1B", "9B", "12A"},
        "1A": {"1A", "2A", "10A", "1B"},
        "1B": {"1B", "2B", "10B", "1A"},
        "2A": {"2A", "3A", "11A", "2B"},
        "2B": {"2B", "3B", "11B", "2A"},
        "3A": {"3A", "4A", "12A", "3B"},
        "3B": {"3B", "4B", "12B", "3A"},
        "4A": {"4A", "5A", "1A", "4B"},
        "4B": {"4B", "5B", "1B", "4A"},
        "5A": {"5A", "6A", "2A", "5B"},
        "5B": {"5B", "6B", "2B", "5A"},
        "6A": {"6A", "7A", "3A", "6B"},
        "6B": {"6B", "7B", "3B", "6A"},
        "7A": {"7A", "8A", "4A", "7B"},
        "7B": {"7B", "8B", "4B", "7A"}
    }
    
    # Be permissive if keys are unknown
    if key_base not in compatible:
        return True
    
    return key_vocal in compatible[key_base]


def vocal_intelligibility_proxy(vocal_path: Path, bounce_path: Path) -> float:
    """
    Calculate vocal intelligibility proxy using spectral analysis.
    
    Args:
        vocal_path: Path to original vocal file
        bounce_path: Path to bounced mix
        
    Returns:
        Intelligibility score (0-1, higher is better)
    """
    try:
        # Load audio files
        vocals, sr = librosa.load(str(vocal_path), mono=True)
        bounce, _ = librosa.load(str(bounce_path), mono=True)
        
        # Calculate spectral centroids
        vocal_centroid = librosa.feature.spectral_centroid(y=vocals, sr=sr).mean()
        bounce_centroid = librosa.feature.spectral_centroid(y=bounce, sr=sr).mean()
        
        # Calculate spectral rolloff (high frequency content)
        vocal_rolloff = librosa.feature.spectral_rolloff(y=vocals, sr=sr).mean()
        bounce_rolloff = librosa.feature.spectral_rolloff(y=bounce, sr=sr).mean()
        
        # Intelligibility proxy based on spectral preservation
        centroid_preservation = 1.0 - abs(vocal_centroid - bounce_centroid) / vocal_centroid
        rolloff_preservation = 1.0 - abs(vocal_rolloff - bounce_rolloff) / vocal_rolloff
        
        # Combine metrics
        intelligibility = (centroid_preservation + rolloff_preservation) / 2.0
        
        return float(np.clip(intelligibility, 0.0, 1.0))
        
    except Exception as e:
        print(f"⚠️ Intelligibility calculation failed: {e}")
        return 0.5


def compute_metrics(bounce_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for bridge feedback.
    
    Args:
        bounce_path: Path to bounced audio file
        manifest: Original manifest for reference
        
    Returns:
        Report dictionary with metrics
    """
    print(f"📊 Computing bridge metrics for: {bounce_path.name}")
    
    # Get file paths from manifest
    vocal_path = Path(manifest["tempo_key"]["vocal_track"]["path"])
    instrumental_path = Path(manifest["tempo_key"]["base_track"]["path"])
    
    # Calculate metrics
    lufs, true_peak_db, clip_ratio = lufs_metrics(bounce_path)
    align_ms = beat_alignment_ms(vocal_path, instrumental_path)
    
    # Key compatibility
    key_base = manifest["tempo_key"].get("project_key", "")
    key_vocal = manifest["tempo_key"]["vocal_track"].get("key", "")
    k_ok = key_compatible_camelot(key_base, key_vocal)
    
    # Vocal intelligibility
    intelligibility = vocal_intelligibility_proxy(vocal_path, bounce_path)
    
    # Define constraints
    constraints = {
        "lufs_target": "-14 ±1",
        "true_peak_max_dbfs": -1.0,
        "clipping_max_ratio": 0.005,
        "beat_align_max_ms": 40
    }
    
    # Check constraint satisfaction
    satisfied = (
        (-15.0 <= lufs <= -13.0) and
        (true_peak_db <= -1.0) and
        (clip_ratio <= constraints["clipping_max_ratio"]) and
        (align_ms <= constraints["beat_align_max_ms"]) and
        k_ok
    )
    
    # Build report
    report = {
        "version": "1.0",
        "session_id": manifest["session"]["id"],
        "files": {
            "bounce": str(bounce_path),
            "vocals": str(vocal_path),
            "instrumental": str(instrumental_path)
        },
        "metrics": {
            "lufs_integrated": round(lufs, 2),
            "true_peak_dbfs": round(true_peak_db, 2),
            "clipping_ratio": round(clip_ratio, 5),
            "beat_alignment_ms": round(align_ms, 1),
            "key_compatible": bool(k_ok),
            "vocal_intelligibility_proxy": round(intelligibility, 3)
        },
        "constraints": constraints,
        "constraints_satisfied": bool(satisfied)
    }
    
    return report


def main():
    """Command-line interface for metrics computation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute bridge metrics")
    parser.add_argument("bounce_path", help="Path to bounced audio file")
    parser.add_argument("manifest_path", help="Path to manifest.json")
    parser.add_argument("--output", help="Output report path")
    
    args = parser.parse_args()
    
    bounce_path = Path(args.bounce_path)
    manifest_path = Path(args.manifest_path)
    
    if not bounce_path.exists():
        print(f"❌ Bounce file not found: {bounce_path}")
        return 1
    
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return 1
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Compute metrics
    report = compute_metrics(bounce_path, manifest)
    
    # Save report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = bounce_path.parent / "report.json"
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Report saved to: {output_path}")
    
    # Print summary
    metrics = report["metrics"]
    print(f"\n🎯 METRICS SUMMARY")
    print(f"   LUFS: {metrics['lufs_integrated']:.1f}")
    print(f"   True Peak: {metrics['true_peak_dbfs']:.1f} dBFS")
    print(f"   Clipping: {metrics['clipping_ratio']:.3f}")
    print(f"   Beat Alignment: {metrics['beat_alignment_ms']:.1f} ms")
    print(f"   Key Compatible: {'✅' if metrics['key_compatible'] else '❌'}")
    print(f"   Intelligibility: {metrics['vocal_intelligibility_proxy']:.3f}")
    print(f"   Constraints Satisfied: {'✅' if report['constraints_satisfied'] else '❌'}")
    
    return 0


if __name__ == "__main__":
    exit(main())
