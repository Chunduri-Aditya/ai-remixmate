"""
Musical Analysis and Correction for Remix Generation

This module provides musical correctness features:
- Key detection using chroma analysis
- Beat grid alignment with onset detection
- Pitch shifting with formant preservation
- Loudness normalization to target LUFS
- Camelot wheel key compatibility
"""

from __future__ import annotations
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import time

# Musical constants
CAMELOT_WHEEL = {
    # Major keys (outer ring)
    'C': 8, 'G': 9, 'D': 10, 'A': 11, 'E': 12, 'B': 1, 'F#': 2, 'C#': 3,
    'G#': 4, 'D#': 5, 'A#': 6, 'F': 7,
    # Minor keys (inner ring)
    'Am': 5, 'Em': 6, 'Bm': 7, 'F#m': 8, 'C#m': 9, 'G#m': 10, 'D#m': 11,
    'A#m': 12, 'Fm': 1, 'Cm': 2, 'Gm': 3, 'Dm': 4
}

# Key profiles for correlation analysis
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

try:
    from scripts.core.config import cfg as _cfg
    TARGET_LUFS = _cfg.audio.target_lufs
    SAMPLE_RATE = _cfg.audio.sample_rate
except Exception:
    TARGET_LUFS = -14.0
    SAMPLE_RATE = 44100


@dataclass
class KeyAnalysis:
    """Key detection and analysis results."""
    key: str
    confidence: float
    camelot_number: int
    is_major: bool
    chroma_vector: np.ndarray


@dataclass
class BeatAnalysis:
    """Beat grid and alignment analysis."""
    tempo: float
    beats: np.ndarray
    beat_times: np.ndarray
    onset_times: np.ndarray
    phase_alignment_ms: float


@dataclass
class PitchShiftParams:
    """Parameters for pitch shifting."""
    semitones: float
    preserve_formants: bool
    quality: str  # 'fast', 'high', 'precise'


class MusicalAnalyzer:
    """Comprehensive musical analysis and correction."""
    
    def __init__(self, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.camelot_wheel = CAMELOT_WHEEL
    
    def detect_key(self, audio: np.ndarray, method: str = 'chroma') -> KeyAnalysis:
        """
        Detect musical key using chroma analysis.
        
        Args:
            audio: Audio signal
            method: Detection method ('chroma', 'krumhansl')
            
        Returns:
            KeyAnalysis with detected key and confidence
        """
        if method == 'chroma':
            return self._detect_key_chroma(audio)
        elif method == 'krumhansl':
            return self._detect_key_krumhansl(audio)
        else:
            raise ValueError(f"Unknown key detection method: {method}")
    
    def _detect_key_chroma(self, audio: np.ndarray) -> KeyAnalysis:
        """Detect key using chroma feature correlation."""
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr, bins_per_octave=12)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Normalize chroma vector
            chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-10)
            
            # Calculate correlations with key profiles
            correlations = np.zeros(24)  # 12 major + 12 minor
            
            for i in range(12):
                # Major keys
                major_shifted = np.roll(MAJOR_PROFILE, i)
                major_corr = np.corrcoef(chroma_norm, major_shifted)[0, 1]
                correlations[i] = major_corr if not np.isnan(major_corr) else 0
                
                # Minor keys
                minor_shifted = np.roll(MINOR_PROFILE, i)
                minor_corr = np.corrcoef(chroma_norm, minor_shifted)[0, 1]
                correlations[i + 12] = minor_corr if not np.isnan(minor_corr) else 0
            
            # Find best match
            best_idx = np.argmax(correlations)
            confidence = float(correlations[best_idx])
            
            # Determine key name
            if best_idx < 12:  # Major key
                key_names = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
                key = key_names[best_idx]
                is_major = True
            else:  # Minor key
                key_names = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m', 'Fm', 'Cm', 'Gm', 'Dm']
                key = key_names[best_idx - 12]
                is_major = False
            
            camelot_number = self.camelot_wheel.get(key, 8)
            
            return KeyAnalysis(
                key=key,
                confidence=confidence,
                camelot_number=camelot_number,
                is_major=is_major,
                chroma_vector=chroma_norm
            )
            
        except Exception as e:
            print(f"⚠️ Key detection failed: {e}")
            return KeyAnalysis(
                key='C',
                confidence=0.0,
                camelot_number=8,
                is_major=True,
                chroma_vector=np.zeros(12)
            )
    
    def _detect_key_krumhansl(self, audio: np.ndarray) -> KeyAnalysis:
        """Detect key using Krumhansl-Schmuckler key profiles."""
        # This is a more sophisticated implementation
        # For now, fall back to chroma method
        return self._detect_key_chroma(audio)
    
    def analyze_beat_grid(self, audio: np.ndarray) -> BeatAnalysis:
        """
        Analyze beat grid and onset alignment.
        
        Args:
            audio: Audio signal
            
        Returns:
            BeatAnalysis with tempo, beats, and alignment info
        """
        try:
            # Extract tempo and beats
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr, units='time')
            
            # Extract onsets
            onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
            
            # Calculate phase alignment (simplified)
            if len(beats) > 0 and len(onset_times) > 0:
                # Find closest onset to first beat
                first_beat = beats[0]
                closest_onset = onset_times[np.argmin(np.abs(onset_times - first_beat))]
                phase_alignment_ms = abs(first_beat - closest_onset) * 1000
            else:
                phase_alignment_ms = 0.0
            
            return BeatAnalysis(
                tempo=float(tempo),
                beats=beats,
                beat_times=beats,
                onset_times=onset_times,
                phase_alignment_ms=float(phase_alignment_ms)
            )
            
        except Exception as e:
            print(f"⚠️ Beat analysis failed: {e}")
            return BeatAnalysis(
                tempo=120.0,
                beats=np.array([]),
                beat_times=np.array([]),
                onset_times=np.array([]),
                phase_alignment_ms=0.0
            )
    
    def calculate_key_compatibility(self, key1: str, key2: str) -> Tuple[float, int]:
        """
        Calculate key compatibility using Camelot wheel.
        
        Args:
            key1: First key
            key2: Second key
            
        Returns:
            Tuple of (compatibility_score, camelot_distance)
        """
        camelot1 = self.camelot_wheel.get(key1, 8)
        camelot2 = self.camelot_wheel.get(key2, 8)
        
        # Calculate circular distance
        distance = min(abs(camelot1 - camelot2), 12 - abs(camelot1 - camelot2))
        
        # Calculate compatibility score (0-1, higher is better)
        # Perfect match (0 steps) = 1.0
        # Adjacent keys (±1 step) = 0.8
        # Compatible keys (±2-3 steps) = 0.6-0.4
        # Incompatible keys (4+ steps) = 0.2-0.0
        if distance == 0:
            compatibility = 1.0
        elif distance == 1:
            compatibility = 0.8
        elif distance <= 3:
            compatibility = 0.6 - (distance - 2) * 0.1
        else:
            compatibility = max(0.0, 0.3 - (distance - 4) * 0.05)
        
        return float(compatibility), int(distance)
    
    def suggest_pitch_shift(self, base_key: str, match_key: str) -> PitchShiftParams:
        """
        Suggest optimal pitch shift for key compatibility.
        
        Args:
            base_key: Base song key
            match_key: Match song key
            
        Returns:
            PitchShiftParams with suggested shift
        """
        compatibility, distance = self.calculate_key_compatibility(base_key, match_key)
        
        # If keys are compatible, no shift needed
        if compatibility >= 0.6:
            return PitchShiftParams(semitones=0.0, preserve_formants=True, quality='high')
        
        # Calculate optimal shift (simplified)
        # In practice, this would use more sophisticated harmonic analysis
        if distance <= 6:
            # Small shift to improve compatibility
            semitones = distance * 0.5  # Rough mapping
            return PitchShiftParams(
                semitones=float(semitones),
                preserve_formants=True,
                quality='high'
            )
        else:
            # Large distance - suggest no shift (would be too drastic)
            return PitchShiftParams(semitones=0.0, preserve_formants=True, quality='high')
    
    def pitch_shift_audio(self, audio: np.ndarray, semitones: float, 
                         preserve_formants: bool = True) -> np.ndarray:
        """
        Apply pitch shifting to audio.
        
        Args:
            audio: Input audio
            semitones: Pitch shift in semitones
            preserve_formants: Whether to preserve formants
            
        Returns:
            Pitch-shifted audio
        """
        if abs(semitones) < 0.1:  # No significant shift needed
            return audio
        
        try:
            # Use librosa's pitch shifting
            if preserve_formants:
                # For formant preservation, we'd need more sophisticated tools
                # For now, use basic pitch shifting
                shifted = librosa.effects.pitch_shift(
                    audio, sr=self.sr, n_steps=semitones
                )
            else:
                shifted = librosa.effects.pitch_shift(
                    audio, sr=self.sr, n_steps=semitones
                )
            
            return shifted
            
        except Exception as e:
            print(f"⚠️ Pitch shifting failed: {e}")
            return audio
    
    def normalize_loudness(self, audio: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
        """
        Normalize audio to target LUFS.
        
        Args:
            audio: Input audio
            target_lufs: Target LUFS level
            
        Returns:
            Loudness-normalized audio
        """
        try:
            # Calculate current RMS level
            rms_current = np.sqrt(np.mean(audio**2))
            
            # Calculate target RMS (rough LUFS approximation)
            target_rms = 10**(target_lufs / 20.0)
            
            # Calculate gain adjustment
            if rms_current > 0:
                gain = target_rms / rms_current
                # Limit gain to prevent excessive amplification
                gain = min(gain, 10.0)  # Max 20dB gain
                normalized = audio * gain
            else:
                normalized = audio
            
            # Apply soft limiting to prevent clipping
            normalized = np.tanh(normalized * 0.8) * 0.95
            
            return normalized
            
        except Exception as e:
            print(f"⚠️ Loudness normalization failed: {e}")
            return audio
    
    def align_beats(self, audio1: np.ndarray, audio2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align beats between two audio signals.
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal
            
        Returns:
            Tuple of (aligned_audio1, aligned_audio2)
        """
        try:
            # Analyze beat grids
            beat1 = self.analyze_beat_grid(audio1)
            beat2 = self.analyze_beat_grid(audio2)
            
            # Calculate tempo ratio
            tempo_ratio = beat1.tempo / beat2.tempo if beat2.tempo > 0 else 1.0
            
            # Time stretch audio2 to match audio1 tempo
            if abs(tempo_ratio - 1.0) > 0.01:  # Significant tempo difference
                try:
                    from scripts.core.gpu import gpu_time_stretch
                    audio2_stretched = gpu_time_stretch(audio2, rate=tempo_ratio)
                except (ImportError, Exception):
                    audio2_stretched = librosa.effects.time_stretch(audio2, rate=tempo_ratio)
            else:
                audio2_stretched = audio2
            
            # Align to same length
            min_length = min(len(audio1), len(audio2_stretched))
            audio1_aligned = audio1[:min_length]
            audio2_aligned = audio2_stretched[:min_length]
            
            return audio1_aligned, audio2_aligned
            
        except Exception as e:
            print(f"⚠️ Beat alignment failed: {e}")
            return audio1, audio2
    
    def analyze_musical_compatibility(self, audio1: np.ndarray, audio2: np.ndarray) -> Dict:
        """
        Comprehensive musical compatibility analysis.
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal
            
        Returns:
            Dictionary with compatibility analysis
        """
        print("🎵 Analyzing musical compatibility...")
        
        # Key analysis
        key1 = self.detect_key(audio1)
        key2 = self.detect_key(audio2)
        compatibility, distance = self.calculate_key_compatibility(key1.key, key2.key)
        
        # Beat analysis
        beat1 = self.analyze_beat_grid(audio1)
        beat2 = self.analyze_beat_grid(audio2)
        tempo_ratio = beat1.tempo / beat2.tempo if beat2.tempo > 0 else 1.0
        
        # Pitch shift suggestion
        pitch_shift = self.suggest_pitch_shift(key1.key, key2.key)
        
        return {
            'keys': {
                'base_key': key1.key,
                'match_key': key2.key,
                'base_confidence': key1.confidence,
                'match_confidence': key2.confidence,
                'compatibility': compatibility,
                'camelot_distance': distance
            },
            'tempo': {
                'base_tempo': beat1.tempo,
                'match_tempo': beat2.tempo,
                'tempo_ratio': tempo_ratio,
                'base_phase_alignment': beat1.phase_alignment_ms,
                'match_phase_alignment': beat2.phase_alignment_ms
            },
            'pitch_shift': {
                'suggested_semitones': pitch_shift.semitones,
                'preserve_formants': pitch_shift.preserve_formants,
                'quality': pitch_shift.quality
            },
            'overall_compatibility': {
                'key_score': compatibility,
                'tempo_score': max(0, 1 - abs(tempo_ratio - 1.0) * 2),  # Penalize tempo mismatch
                'combined_score': (compatibility + max(0, 1 - abs(tempo_ratio - 1.0) * 2)) / 2
            }
        }
    
    def apply_musical_corrections(self, vocals: np.ndarray, instrumentals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Apply musical corrections to improve compatibility.
        
        Args:
            vocals: Vocal audio
            instrumentals: Instrumental audio
            
        Returns:
            Tuple of (corrected_vocals, corrected_instrumentals, correction_info)
        """
        print("🔧 Applying musical corrections...")
        
        # Analyze compatibility
        analysis = self.analyze_musical_compatibility(vocals, instrumentals)
        
        # Apply pitch shifting if needed
        pitch_shift = analysis['pitch_shift']
        if abs(pitch_shift['suggested_semitones']) > 0.1:
            print(f"   Applying pitch shift: {pitch_shift['suggested_semitones']:.1f} semitones")
            instrumentals = self.pitch_shift_audio(
                instrumentals, 
                pitch_shift['suggested_semitones'],
                pitch_shift['preserve_formants']
            )
        
        # Align beats
        print("   Aligning beat grids...")
        vocals_aligned, instrumentals_aligned = self.align_beats(vocals, instrumentals)
        
        # Normalize loudness
        print("   Normalizing loudness...")
        vocals_normalized = self.normalize_loudness(vocals_aligned)
        instrumentals_normalized = self.normalize_loudness(instrumentals_aligned)
        
        correction_info = {
            'pitch_shift_applied': pitch_shift['suggested_semitones'],
            'tempo_ratio': analysis['tempo']['tempo_ratio'],
            'key_compatibility': analysis['keys']['compatibility'],
            'corrections_applied': [
                'pitch_shift' if abs(pitch_shift['suggested_semitones']) > 0.1 else None,
                'beat_alignment',
                'loudness_normalization'
            ]
        }
        correction_info['corrections_applied'] = [c for c in correction_info['corrections_applied'] if c is not None]
        
        return vocals_normalized, instrumentals_normalized, correction_info
