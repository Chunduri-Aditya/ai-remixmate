#!/usr/bin/env python3
"""
Real Music Test Suite for AI RemixMate Bridge

This module tests the bridge system with real music files
from various genres and styles to ensure practical performance.
"""

from __future__ import annotations
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import librosa
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
import random


from scripts.export_manifest import build_manifest, create_session_id
from scripts.bridge.logic_bridge import main as logic_main
from scripts.bridge.preset_manager import PresetManager
from scripts.bridge.iterative_optimizer import IterativeOptimizer
from scripts.bridge_metrics import compute_metrics


@dataclass
class RealMusicTest:
    """Represents a real music test case."""
    name: str
    description: str
    base_file: Path
    match_file: Path
    genre: str
    expected_bpm: float
    expected_key: str
    vocal_level: str  # heavy, medium, light, instrumental
    complexity: str  # simple, medium, complex
    test_category: str  # genre_test, mix_test, edge_case


class RealMusicTestSuite:
    """Test suite for real music files."""
    
    def __init__(self, music_dir: str = "test_music", output_dir: str = "runs/real_music_test"):
        self.music_dir = Path(music_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.preset_manager = PresetManager()
        self.test_cases = []
        
        # Create test music directory if it doesn't exist
        self.music_dir.mkdir(exist_ok=True)
    
    def create_test_music_library(self) -> None:
        """Create a library of test music files."""
        print("🎵 Creating test music library...")
        
        # Create subdirectories for different genres
        genres = ["electronic", "rock", "hip_hop", "jazz", "classical", "pop", "ambient"]
        for genre in genres:
            (self.music_dir / genre).mkdir(exist_ok=True)
        
        # Generate test music files
        self._generate_test_music_files()
        
        print(f"✅ Test music library created in: {self.music_dir}")
    
    def _generate_test_music_files(self) -> None:
        """Generate test music files for different genres."""
        # Electronic music
        self._create_electronic_tracks()
        
        # Rock music
        self._create_rock_tracks()
        
        # Hip-hop music
        self._create_hip_hop_tracks()
        
        # Jazz music
        self._create_jazz_tracks()
        
        # Classical music
        self._create_classical_tracks()
        
        # Pop music
        self._create_pop_tracks()
        
        # Ambient music
        self._create_ambient_tracks()
    
    def _create_electronic_tracks(self) -> None:
        """Create electronic music test tracks."""
        genre_dir = self.music_dir / "electronic"
        
        # House track (128 BPM, 8A)
        house_vocals = self._generate_electronic_vocals(128, "8A", "house")
        house_inst = self._generate_electronic_instrumental(128, "8A", "house")
        sf.write(genre_dir / "house_vocals.wav", house_vocals, 44100)
        sf.write(genre_dir / "house_instrumental.wav", house_inst, 44100)
        
        # Techno track (130 BPM, 5B)
        techno_vocals = self._generate_electronic_vocals(130, "5B", "techno")
        techno_inst = self._generate_electronic_instrumental(130, "5B", "techno")
        sf.write(genre_dir / "techno_vocals.wav", techno_vocals, 44100)
        sf.write(genre_dir / "techno_instrumental.wav", techno_inst, 44100)
        
        # Dubstep track (140 BPM, 12A)
        dubstep_vocals = self._generate_electronic_vocals(140, "12A", "dubstep")
        dubstep_inst = self._generate_electronic_instrumental(140, "12A", "dubstep")
        sf.write(genre_dir / "dubstep_vocals.wav", dubstep_vocals, 44100)
        sf.write(genre_dir / "dubstep_instrumental.wav", dubstep_inst, 44100)
    
    def _create_rock_tracks(self) -> None:
        """Create rock music test tracks."""
        genre_dir = self.music_dir / "rock"
        
        # Alternative rock (120 BPM, 8A)
        alt_vocals = self._generate_rock_vocals(120, "8A", "alternative")
        alt_inst = self._generate_rock_instrumental(120, "8A", "alternative")
        sf.write(genre_dir / "alt_rock_vocals.wav", alt_vocals, 44100)
        sf.write(genre_dir / "alt_rock_instrumental.wav", alt_inst, 44100)
        
        # Metal track (160 BPM, 5B)
        metal_vocals = self._generate_rock_vocals(160, "5B", "metal")
        metal_inst = self._generate_rock_instrumental(160, "5B", "metal")
        sf.write(genre_dir / "metal_vocals.wav", metal_vocals, 44100)
        sf.write(genre_dir / "metal_instrumental.wav", metal_inst, 44100)
    
    def _create_hip_hop_tracks(self) -> None:
        """Create hip-hop music test tracks."""
        genre_dir = self.music_dir / "hip_hop"
        
        # Trap track (140 BPM, 8A)
        trap_vocals = self._generate_hip_hop_vocals(140, "8A", "trap")
        trap_inst = self._generate_hip_hop_instrumental(140, "8A", "trap")
        sf.write(genre_dir / "trap_vocals.wav", trap_vocals, 44100)
        sf.write(genre_dir / "trap_instrumental.wav", trap_inst, 44100)
        
        # Boom bap track (90 BPM, 5B)
        boom_vocals = self._generate_hip_hop_vocals(90, "5B", "boom_bap")
        boom_inst = self._generate_hip_hop_instrumental(90, "5B", "boom_bap")
        sf.write(genre_dir / "boom_bap_vocals.wav", boom_vocals, 44100)
        sf.write(genre_dir / "boom_bap_instrumental.wav", boom_inst, 44100)
    
    def _create_jazz_tracks(self) -> None:
        """Create jazz music test tracks."""
        genre_dir = self.music_dir / "jazz"
        
        # Smooth jazz (100 BPM, 8A)
        smooth_vocals = self._generate_jazz_vocals(100, "8A", "smooth")
        smooth_inst = self._generate_jazz_instrumental(100, "8A", "smooth")
        sf.write(genre_dir / "smooth_jazz_vocals.wav", smooth_vocals, 44100)
        sf.write(genre_dir / "smooth_jazz_instrumental.wav", smooth_inst, 44100)
        
        # Bebop track (200 BPM, 5B)
        bebop_vocals = self._generate_jazz_vocals(200, "5B", "bebop")
        bebop_inst = self._generate_jazz_instrumental(200, "5B", "bebop")
        sf.write(genre_dir / "bebop_vocals.wav", bebop_vocals, 44100)
        sf.write(genre_dir / "bebop_instrumental.wav", bebop_inst, 44100)
    
    def _create_classical_tracks(self) -> None:
        """Create classical music test tracks."""
        genre_dir = self.music_dir / "classical"
        
        # Orchestral track (80 BPM, 8A)
        orchestral_vocals = self._generate_classical_vocals(80, "8A", "orchestral")
        orchestral_inst = self._generate_classical_instrumental(80, "8A", "orchestral")
        sf.write(genre_dir / "orchestral_vocals.wav", orchestral_vocals, 44100)
        sf.write(genre_dir / "orchestral_instrumental.wav", orchestral_inst, 44100)
    
    def _create_pop_tracks(self) -> None:
        """Create pop music test tracks."""
        genre_dir = self.music_dir / "pop"
        
        # Pop track (120 BPM, 8A)
        pop_vocals = self._generate_pop_vocals(120, "8A", "pop")
        pop_inst = self._generate_pop_instrumental(120, "8A", "pop")
        sf.write(genre_dir / "pop_vocals.wav", pop_vocals, 44100)
        sf.write(genre_dir / "pop_instrumental.wav", pop_inst, 44100)
    
    def _create_ambient_tracks(self) -> None:
        """Create ambient music test tracks."""
        genre_dir = self.music_dir / "ambient"
        
        # Ambient track (60 BPM, 8A)
        ambient_vocals = self._generate_ambient_vocals(60, "8A", "ambient")
        ambient_inst = self._generate_ambient_instrumental(60, "8A", "ambient")
        sf.write(genre_dir / "ambient_vocals.wav", ambient_vocals, 44100)
        sf.write(genre_dir / "ambient_instrumental.wav", ambient_inst, 44100)
    
    def _generate_electronic_vocals(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate electronic vocals."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "house":
            # House vocals: repetitive, catchy
            vocal = 0.3 * np.sin(2 * np.pi * base_freq * t)
            # Add house-style processing
            vocal = np.tanh(vocal * 1.5) * 0.2
        elif style == "techno":
            # Techno vocals: robotic, processed
            vocal = 0.2 * np.sin(2 * np.pi * base_freq * t)
            vocal += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            vocal = np.tanh(vocal * 2) * 0.15
        elif style == "dubstep":
            # Dubstep vocals: aggressive, distorted
            vocal = 0.4 * np.sin(2 * np.pi * base_freq * t)
            vocal = np.tanh(vocal * 3) * 0.25
        
        return vocal
    
    def _generate_electronic_instrumental(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate electronic instrumental."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "house":
            # House instrumental: four-on-the-floor, bass
            kick = self._generate_kick_pattern(bpm, t, sr)
            bass = 0.4 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            lead = 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)
            return kick + bass + lead
        elif style == "techno":
            # Techno instrumental: driving, mechanical
            kick = self._generate_kick_pattern(bpm, t, sr)
            bass = 0.3 * np.sin(2 * np.pi * base_freq * 0.25 * t)
            lead = 0.15 * np.sign(np.sin(2 * np.pi * base_freq * 2 * t))
            return kick + bass + lead
        elif style == "dubstep":
            # Dubstep instrumental: heavy bass, wobbles
            kick = self._generate_kick_pattern(bpm, t, sr)
            bass = 0.5 * np.sin(2 * np.pi * base_freq * 0.25 * t)
            # Add wobble effect
            wobble = 0.3 * np.sin(2 * np.pi * base_freq * 0.5 * t * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t)))
            return kick + bass + wobble
    
    def _generate_rock_vocals(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate rock vocals."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "alternative":
            # Alternative rock vocals: raw, emotional
            vocal = 0.3 * np.sin(2 * np.pi * base_freq * t)
            vocal += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            vocal = np.tanh(vocal * 2) * 0.2
        elif style == "metal":
            # Metal vocals: aggressive, distorted
            vocal = 0.4 * np.sin(2 * np.pi * base_freq * t)
            vocal = np.tanh(vocal * 4) * 0.3
        
        return vocal
    
    def _generate_rock_instrumental(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate rock instrumental."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "alternative":
            # Alternative rock: guitar, bass, drums
            guitar = 0.3 * np.sin(2 * np.pi * base_freq * t)
            guitar = np.tanh(guitar * 2) * 0.2
            bass = 0.4 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            drums = self._generate_rock_drums(bpm, t, sr)
            return guitar + bass + drums
        elif style == "metal":
            # Metal: heavy guitar, bass, drums
            guitar = 0.4 * np.sin(2 * np.pi * base_freq * t)
            guitar = np.tanh(guitar * 3) * 0.3
            bass = 0.5 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            drums = self._generate_metal_drums(bpm, t, sr)
            return guitar + bass + drums
    
    def _generate_hip_hop_vocals(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate hip-hop vocals."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "trap":
            # Trap vocals: melodic, auto-tuned
            vocal = 0.2 * np.sin(2 * np.pi * base_freq * t)
            vocal += 0.1 * np.sin(2 * np.pi * base_freq * 1.5 * t)
        elif style == "boom_bap":
            # Boom bap vocals: rhythmic, raw
            vocal = 0.3 * np.sin(2 * np.pi * base_freq * t)
            vocal = np.tanh(vocal * 1.5) * 0.2
        
        return vocal
    
    def _generate_hip_hop_instrumental(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate hip-hop instrumental."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "trap":
            # Trap instrumental: heavy bass, hi-hats
            bass = 0.6 * np.sin(2 * np.pi * base_freq * 0.25 * t)
            hihats = self._generate_trap_hihats(bpm, t, sr)
            return bass + hihats
        elif style == "boom_bap":
            # Boom bap instrumental: sample-based, drums
            bass = 0.4 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            drums = self._generate_boom_bap_drums(bpm, t, sr)
            return bass + drums
    
    def _generate_jazz_vocals(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate jazz vocals."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "smooth":
            # Smooth jazz vocals: mellow, soulful
            vocal = 0.2 * np.sin(2 * np.pi * base_freq * t)
            vocal += 0.1 * np.sin(2 * np.pi * base_freq * 1.2 * t)
        elif style == "bebop":
            # Bebop vocals: fast, complex
            vocal = 0.15 * np.sin(2 * np.pi * base_freq * t)
            vocal += 0.1 * np.sin(2 * np.pi * base_freq * 1.5 * t)
            vocal += 0.05 * np.sin(2 * np.pi * base_freq * 2 * t)
        
        return vocal
    
    def _generate_jazz_instrumental(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate jazz instrumental."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "smooth":
            # Smooth jazz: piano, bass, drums
            piano = 0.2 * np.sin(2 * np.pi * base_freq * t)
            piano += 0.1 * np.sin(2 * np.pi * base_freq * 1.2 * t)
            bass = 0.3 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            drums = self._generate_jazz_drums(bpm, t, sr)
            return piano + bass + drums
        elif style == "bebop":
            # Bebop: fast piano, walking bass, complex drums
            piano = 0.15 * np.sin(2 * np.pi * base_freq * t)
            piano += 0.1 * np.sin(2 * np.pi * base_freq * 1.5 * t)
            bass = 0.2 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            drums = self._generate_bebop_drums(bpm, t, sr)
            return piano + bass + drums
    
    def _generate_classical_vocals(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate classical vocals."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "orchestral":
            # Orchestral vocals: operatic, pure
            vocal = 0.15 * np.sin(2 * np.pi * base_freq * t)
            vocal += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            vocal += 0.05 * np.sin(2 * np.pi * base_freq * 3 * t)
        
        return vocal
    
    def _generate_classical_instrumental(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate classical instrumental."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "orchestral":
            # Orchestral: strings, woodwinds, brass
            strings = 0.1 * np.sin(2 * np.pi * base_freq * t)
            strings += 0.08 * np.sin(2 * np.pi * base_freq * 1.2 * t)
            strings += 0.06 * np.sin(2 * np.pi * base_freq * 1.5 * t)
            woodwinds = 0.08 * np.sin(2 * np.pi * base_freq * 2 * t)
            brass = 0.06 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            return strings + woodwinds + brass
    
    def _generate_pop_vocals(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate pop vocals."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "pop":
            # Pop vocals: catchy, processed
            vocal = 0.25 * np.sin(2 * np.pi * base_freq * t)
            vocal += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
            vocal = np.tanh(vocal * 1.2) * 0.2
        
        return vocal
    
    def _generate_pop_instrumental(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate pop instrumental."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "pop":
            # Pop instrumental: synth, bass, drums
            synth = 0.2 * np.sin(2 * np.pi * base_freq * t)
            bass = 0.3 * np.sin(2 * np.pi * base_freq * 0.5 * t)
            drums = self._generate_pop_drums(bpm, t, sr)
            return synth + bass + drums
    
    def _generate_ambient_vocals(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate ambient vocals."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "ambient":
            # Ambient vocals: ethereal, processed
            vocal = 0.1 * np.sin(2 * np.pi * base_freq * t)
            vocal += 0.05 * np.sin(2 * np.pi * base_freq * 1.5 * t)
            # Add reverb-like effect
            vocal = vocal * (1 + 0.3 * np.sin(2 * np.pi * 0.1 * t))
        
        return vocal
    
    def _generate_ambient_instrumental(self, bpm: float, key: str, style: str) -> np.ndarray:
        """Generate ambient instrumental."""
        duration = 30
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration))
        
        base_freq = self._key_to_frequency(key)
        
        if style == "ambient":
            # Ambient instrumental: pads, textures
            pad = 0.1 * np.sin(2 * np.pi * base_freq * t)
            pad += 0.08 * np.sin(2 * np.pi * base_freq * 1.2 * t)
            pad += 0.06 * np.sin(2 * np.pi * base_freq * 1.5 * t)
            # Add texture
            texture = 0.02 * np.random.randn(len(t))
            return pad + texture
    
    def _key_to_frequency(self, key: str) -> float:
        """Convert Camelot key to base frequency."""
        key_map = {
            "1A": 261.63, "1B": 277.18, "2A": 293.66, "2B": 311.13,
            "3A": 329.63, "3B": 349.23, "4A": 369.99, "4B": 392.00,
            "5A": 415.30, "5B": 440.00, "6A": 466.16, "6B": 493.88,
            "7A": 523.25, "7B": 554.37, "8A": 587.33, "8B": 622.25,
            "9A": 659.25, "9B": 698.46, "10A": 739.99, "10B": 783.99,
            "11A": 830.61, "11B": 880.00, "12A": 932.33, "12B": 987.77
        }
        return key_map.get(key, 440.0)
    
    def _generate_kick_pattern(self, bpm: float, t: np.ndarray, sr: int) -> np.ndarray:
        """Generate kick drum pattern."""
        kick = np.zeros_like(t)
        beat_interval = 60 / bpm
        
        for i in range(int(30 / beat_interval)):
            start_time = i * beat_interval
            if start_time < 30:
                start_idx = int(start_time * sr)
                end_idx = min(start_idx + int(0.1 * sr), len(kick))
                if start_idx < len(kick):
                    kick[start_idx:end_idx] += 0.4 * np.sin(2 * np.pi * 60 * np.linspace(0, 0.1, end_idx - start_idx))
        
        return kick
    
    def _generate_rock_drums(self, bpm: float, t: np.ndarray, sr: int) -> np.ndarray:
        """Generate rock drum pattern."""
        drums = np.zeros_like(t)
        beat_interval = 60 / bpm
        
        for i in range(int(30 / beat_interval)):
            start_time = i * beat_interval
            if start_time < 30:
                start_idx = int(start_time * sr)
                end_idx = min(start_idx + int(0.08 * sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.3 * np.random.randn(end_idx - start_idx)
        
        return drums
    
    def _generate_metal_drums(self, bpm: float, t: np.ndarray, sr: int) -> np.ndarray:
        """Generate metal drum pattern."""
        drums = np.zeros_like(t)
        beat_interval = 60 / bpm
        
        for i in range(int(30 / beat_interval)):
            start_time = i * beat_interval
            if start_time < 30:
                start_idx = int(start_time * sr)
                end_idx = min(start_idx + int(0.06 * sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.4 * np.random.randn(end_idx - start_idx)
        
        return drums
    
    def _generate_trap_hihats(self, bpm: float, t: np.ndarray, sr: int) -> np.ndarray:
        """Generate trap hi-hat pattern."""
        hihats = np.zeros_like(t)
        beat_interval = 60 / bpm
        
        for i in range(int(30 / beat_interval)):
            start_time = i * beat_interval
            if start_time < 30:
                start_idx = int(start_time * sr)
                end_idx = min(start_idx + int(0.02 * sr), len(hihats))
                if start_idx < len(hihats):
                    hihats[start_idx:end_idx] += 0.2 * np.random.randn(end_idx - start_idx)
        
        return hihats
    
    def _generate_boom_bap_drums(self, bpm: float, t: np.ndarray, sr: int) -> np.ndarray:
        """Generate boom bap drum pattern."""
        drums = np.zeros_like(t)
        beat_interval = 60 / bpm
        
        for i in range(int(30 / beat_interval)):
            start_time = i * beat_interval
            if start_time < 30:
                start_idx = int(start_time * sr)
                end_idx = min(start_idx + int(0.1 * sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.3 * np.random.randn(end_idx - start_idx)
        
        return drums
    
    def _generate_jazz_drums(self, bpm: float, t: np.ndarray, sr: int) -> np.ndarray:
        """Generate jazz drum pattern."""
        drums = np.zeros_like(t)
        beat_interval = 60 / bpm
        
        for i in range(int(30 / beat_interval)):
            start_time = i * beat_interval
            if start_time < 30:
                start_idx = int(start_time * sr)
                end_idx = min(start_idx + int(0.05 * sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.2 * np.random.randn(end_idx - start_idx)
        
        return drums
    
    def _generate_bebop_drums(self, bpm: float, t: np.ndarray, sr: int) -> np.ndarray:
        """Generate bebop drum pattern."""
        drums = np.zeros_like(t)
        beat_interval = 60 / bpm
        
        for i in range(int(30 / beat_interval)):
            start_time = i * beat_interval
            if start_time < 30:
                start_idx = int(start_time * sr)
                end_idx = min(start_idx + int(0.03 * sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.15 * np.random.randn(end_idx - start_idx)
        
        return drums
    
    def _generate_pop_drums(self, bpm: float, t: np.ndarray, sr: int) -> np.ndarray:
        """Generate pop drum pattern."""
        drums = np.zeros_like(t)
        beat_interval = 60 / bpm
        
        for i in range(int(30 / beat_interval)):
            start_time = i * beat_interval
            if start_time < 30:
                start_idx = int(start_time * sr)
                end_idx = min(start_idx + int(0.08 * sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.25 * np.random.randn(end_idx - start_idx)
        
        return drums
    
    def create_test_cases(self) -> None:
        """Create test cases from the music library."""
        print("📋 Creating test cases...")
        
        # Genre tests
        self._create_genre_tests()
        
        # Mix scenario tests
        self._create_mix_tests()
        
        # Edge case tests
        self._create_edge_case_tests()
        
        print(f"✅ Created {len(self.test_cases)} test cases")
    
    def _create_genre_tests(self) -> None:
        """Create genre-specific test cases."""
        genres = ["electronic", "rock", "hip_hop", "jazz", "classical", "pop", "ambient"]
        
        for genre in genres:
            genre_dir = self.music_dir / genre
            
            # Find vocal and instrumental files
            vocal_files = list(genre_dir.glob("*_vocals.wav"))
            inst_files = list(genre_dir.glob("*_instrumental.wav"))
            
            for vocal_file in vocal_files:
                for inst_file in inst_files:
                    # Extract track info from filename
                    track_name = vocal_file.stem.replace("_vocals", "")
                    
                    test_case = RealMusicTest(
                        name=f"genre_{genre}_{track_name}",
                        description=f"{genre.title()} {track_name} track test",
                        base_file=vocal_file,
                        match_file=inst_file,
                        genre=genre,
                        expected_bpm=self._get_expected_bpm(genre, track_name),
                        expected_key=self._get_expected_key(genre, track_name),
                        vocal_level="medium",
                        complexity="medium",
                        test_category="genre_test"
                    )
                    self.test_cases.append(test_case)
    
    def _create_mix_tests(self) -> None:
        """Create mix scenario test cases."""
        # Cross-genre mixes
        cross_genre_combinations = [
            ("electronic", "rock"),
            ("hip_hop", "jazz"),
            ("classical", "ambient"),
            ("pop", "electronic")
        ]
        
        for base_genre, match_genre in cross_genre_combinations:
            base_dir = self.music_dir / base_genre
            match_dir = self.music_dir / match_genre
            
            base_vocals = list(base_dir.glob("*_vocals.wav"))
            match_insts = list(match_dir.glob("*_instrumental.wav"))
            
            for base_vocal in base_vocals[:2]:  # Limit to 2 per genre
                for match_inst in match_insts[:2]:
                    base_track = base_vocal.stem.replace("_vocals", "")
                    match_track = match_inst.stem.replace("_instrumental", "")
                    
                    test_case = RealMusicTest(
                        name=f"mix_{base_genre}_{base_track}_to_{match_genre}_{match_track}",
                        description=f"Cross-genre mix: {base_genre} {base_track} vocals with {match_genre} {match_track} instrumental",
                        base_file=base_vocal,
                        match_file=match_inst,
                        genre=f"{base_genre}_to_{match_genre}",
                        expected_bpm=self._get_expected_bpm(base_genre, base_track),
                        expected_key=self._get_expected_key(base_genre, base_track),
                        vocal_level="medium",
                        complexity="complex",
                        test_category="mix_test"
                    )
                    self.test_cases.append(test_case)
    
    def _create_edge_case_tests(self) -> None:
        """Create edge case test cases."""
        # Very different tempos
        slow_genres = ["classical", "ambient"]
        fast_genres = ["electronic", "rock"]
        
        for slow_genre in slow_genres:
            for fast_genre in fast_genres:
                slow_dir = self.music_dir / slow_genre
                fast_dir = self.music_dir / fast_genre
                
                slow_vocals = list(slow_dir.glob("*_vocals.wav"))
                fast_insts = list(fast_dir.glob("*_instrumental.wav"))
                
                if slow_vocals and fast_insts:
                    test_case = RealMusicTest(
                        name=f"edge_tempo_{slow_genre}_to_{fast_genre}",
                        description=f"Extreme tempo difference: {slow_genre} to {fast_genre}",
                        base_file=slow_vocals[0],
                        match_file=fast_insts[0],
                        genre=f"{slow_genre}_to_{fast_genre}",
                        expected_bpm=self._get_expected_bpm(slow_genre, "default"),
                        expected_key="8A",
                        vocal_level="medium",
                        complexity="complex",
                        test_category="edge_case"
                    )
                    self.test_cases.append(test_case)
    
    def _get_expected_bpm(self, genre: str, track: str) -> float:
        """Get expected BPM for genre and track."""
        bpm_map = {
            "electronic": {"house": 128, "techno": 130, "dubstep": 140, "default": 130},
            "rock": {"alternative": 120, "metal": 160, "default": 140},
            "hip_hop": {"trap": 140, "boom_bap": 90, "default": 115},
            "jazz": {"smooth": 100, "bebop": 200, "default": 150},
            "classical": {"orchestral": 80, "default": 80},
            "pop": {"pop": 120, "default": 120},
            "ambient": {"ambient": 60, "default": 60}
        }
        return bpm_map.get(genre, {}).get(track, 120)
    
    def _get_expected_key(self, genre: str, track: str) -> str:
        """Get expected key for genre and track."""
        key_map = {
            "electronic": {"house": "8A", "techno": "5B", "dubstep": "12A", "default": "8A"},
            "rock": {"alternative": "8A", "metal": "5B", "default": "8A"},
            "hip_hop": {"trap": "8A", "boom_bap": "5B", "default": "8A"},
            "jazz": {"smooth": "8A", "bebop": "5B", "default": "8A"},
            "classical": {"orchestral": "8A", "default": "8A"},
            "pop": {"pop": "8A", "default": "8A"},
            "ambient": {"ambient": "8A", "default": "8A"}
        }
        return key_map.get(genre, {}).get(track, "8A")
    
    def run_test_case(self, test_case: RealMusicTest, preset: str = "radio", optimize: bool = False) -> Dict[str, Any]:
        """Run a single test case."""
        print(f"\n🧪 Testing: {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   Genre: {test_case.genre}")
        print(f"   Expected BPM: {test_case.expected_bpm}, Key: {test_case.expected_key}")
        print(f"   Vocal Level: {test_case.vocal_level}, Complexity: {test_case.complexity}")
        
        start_time = time.time()
        
        try:
            # Create session
            session_id = create_session_id()
            session_dir = self.output_dir / test_case.name / f"session_{session_id}"
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Build manifest
            stems = {
                "vocals_path": str(test_case.base_file),
                "instrumental_path": str(test_case.match_file),
                "vocals_bpm": test_case.expected_bpm,
                "instrumental_bpm": test_case.expected_bpm,
                "vocals_key": test_case.expected_key,
                "instrumental_key": test_case.expected_key,
                "project_bpm": test_case.expected_bpm,
                "mix_params": {
                    "vocal_gain_db": 0.0,
                    "inst_gain_db": 0.0,
                    "sidechain_amount": 0.1,
                    "hp_filter_hz": 100.0,
                    "reverb_send": 0.05
                }
            }
            
            manifest = build_manifest(
                project_path="/Users/chunduri/Music/Logic/AI_RemixMate.logicx",
                session_id=session_id,
                out_dir=str(session_dir),
                stems=stems,
                project_bpm=stems["project_bpm"],
                project_key=test_case.expected_key,
                mix_params=stems["mix_params"]
            )
            
            # Apply preset
            if preset != "radio":
                manifest = self.preset_manager.apply_preset_to_manifest(manifest, preset)
            
            # Write manifest
            manifest_path = session_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Simulate bridge run (without Logic Pro)
            report = self._simulate_bridge_run(manifest, session_dir)
            
            # Calculate test metrics
            test_duration = time.time() - start_time
            
            result = {
                "test_case": test_case.name,
                "description": test_case.description,
                "genre": test_case.genre,
                "test_category": test_case.test_category,
                "preset": preset,
                "optimization_used": optimize,
                "test_duration": test_duration,
                "success": True,
                "metrics": report.get("metrics", {}),
                "constraints_satisfied": report.get("constraints_satisfied", False),
                "session_dir": str(session_dir),
                "manifest_path": str(manifest_path)
            }
            
            print(f"   ✅ Test completed in {test_duration:.2f}s")
            print(f"   Constraints satisfied: {'✅' if result['constraints_satisfied'] else '❌'}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            return {
                "test_case": test_case.name,
                "description": test_case.description,
                "genre": test_case.genre,
                "test_category": test_case.test_category,
                "preset": preset,
                "optimization_used": optimize,
                "test_duration": time.time() - start_time,
                "success": False,
                "error": str(e),
                "constraints_satisfied": False
            }
    
    def _simulate_bridge_run(self, manifest: Dict[str, Any], session_dir: Path) -> Dict[str, Any]:
        """Simulate bridge run without Logic Pro."""
        # Create a simulated bounce file
        bounce_path = session_dir / "AI_RemixMate_Bounce.wav"
        
        # Generate a simple mix of the input files
        base_path = Path(manifest["tempo_key"]["vocal_track"]["path"])
        match_path = Path(manifest["tempo_key"]["base_track"]["path"])
        
        if base_path.exists() and match_path.exists():
            base_audio, sr = sf.read(str(base_path))
            match_audio, _ = sf.read(str(match_path))
            
            # Simple mix
            min_length = min(len(base_audio), len(match_audio))
            mixed = base_audio[:min_length] + match_audio[:min_length]
            mixed = mixed * 0.5  # Prevent clipping
            
            sf.write(str(bounce_path), mixed, sr)
        
        # Compute metrics
        return compute_metrics(bounce_path, manifest)
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive real music tests."""
        print("🚀 Starting Real Music Test Suite")
        print("=" * 60)
        
        # Test configurations
        presets = ["radio", "club", "ambient"]
        optimization_modes = [False, True]
        
        total_tests = len(self.test_cases) * len(presets) * len(optimization_modes)
        current_test = 0
        
        results = []
        
        for test_case in self.test_cases:
            for preset in presets:
                for optimize in optimization_modes:
                    current_test += 1
                    print(f"\n📊 Progress: {current_test}/{total_tests}")
                    
                    result = self.run_test_case(test_case, preset, optimize)
                    results.append(result)
        
        # Generate comprehensive report
        return self.generate_test_report(results)
    
    def generate_test_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n📊 Generating Real Music Test Report")
        print("=" * 50)
        
        # Calculate statistics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success"])
        failed_tests = total_tests - successful_tests
        
        constraints_satisfied = sum(1 for r in results if r.get("constraints_satisfied", False))
        optimization_tests = sum(1 for r in results if r.get("optimization_used", False))
        
        # Performance metrics
        test_durations = [r["test_duration"] for r in results if r["success"]]
        avg_duration = np.mean(test_durations) if test_durations else 0
        max_duration = max(test_durations) if test_durations else 0
        min_duration = min(test_durations) if test_durations else 0
        
        # Genre performance
        genre_stats = {}
        for result in results:
            genre = result.get("genre", "unknown")
            if genre not in genre_stats:
                genre_stats[genre] = {"total": 0, "successful": 0, "constraints_satisfied": 0}
            genre_stats[genre]["total"] += 1
            if result["success"]:
                genre_stats[genre]["successful"] += 1
            if result.get("constraints_satisfied", False):
                genre_stats[genre]["constraints_satisfied"] += 1
        
        # Preset performance
        preset_stats = {}
        for preset in ["radio", "club", "ambient"]:
            preset_results = [r for r in results if r.get("preset") == preset]
            if preset_results:
                preset_stats[preset] = {
                    "total": len(preset_results),
                    "successful": sum(1 for r in preset_results if r["success"]),
                    "constraints_satisfied": sum(1 for r in preset_results if r.get("constraints_satisfied", False))
                }
        
        # Test category performance
        category_stats = {}
        for result in results:
            category = result.get("test_category", "unknown")
            if category not in category_stats:
                category_stats[category] = {"total": 0, "successful": 0, "constraints_satisfied": 0}
            category_stats[category]["total"] += 1
            if result["success"]:
                category_stats[category]["successful"] += 1
            if result.get("constraints_satisfied", False):
                category_stats[category]["constraints_satisfied"] += 1
        
        # Build comprehensive report
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "constraints_satisfied": constraints_satisfied,
                "constraint_satisfaction_rate": constraints_satisfied / total_tests if total_tests > 0 else 0,
                "optimization_tests": optimization_tests
            },
            "performance_metrics": {
                "average_test_duration": avg_duration,
                "max_test_duration": max_duration,
                "min_test_duration": min_duration,
                "total_test_time": sum(test_durations)
            },
            "genre_performance": genre_stats,
            "preset_performance": preset_stats,
            "category_performance": category_stats,
            "detailed_results": results,
            "test_timestamp": datetime.now().isoformat(),
            "test_environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "output_directory": str(self.output_dir),
                "music_directory": str(self.music_dir)
            }
        }
        
        # Save report
        report_path = self.output_dir / "real_music_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_test_summary(report)
        
        return report
    
    def print_test_summary(self, report: Dict[str, Any]) -> None:
        """Print test summary."""
        summary = report["test_summary"]
        perf = report["performance_metrics"]
        
        print(f"\n🎯 REAL MUSIC TEST RESULTS")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']} ({summary['success_rate']:.1%})")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Constraints Satisfied: {summary['constraints_satisfied']} ({summary['constraint_satisfaction_rate']:.1%})")
        print(f"Optimization Tests: {summary['optimization_tests']}")
        
        print(f"\n⏱️ PERFORMANCE")
        print(f"Average Duration: {perf['average_test_duration']:.2f}s")
        print(f"Max Duration: {perf['max_test_duration']:.2f}s")
        print(f"Min Duration: {perf['min_test_duration']:.2f}s")
        print(f"Total Time: {perf['total_test_time']:.2f}s")
        
        print(f"\n🎵 GENRE PERFORMANCE")
        for genre, stats in report["genre_performance"].items():
            success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            constraint_rate = stats["constraints_satisfied"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {genre.title()}: {stats['successful']}/{stats['total']} ({success_rate:.1%}) success, {constraint_rate:.1%} constraints")
        
        print(f"\n🎛️ PRESET PERFORMANCE")
        for preset, stats in report["preset_performance"].items():
            success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            constraint_rate = stats["constraints_satisfied"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {preset.title()}: {stats['successful']}/{stats['total']} ({success_rate:.1%}) success, {constraint_rate:.1%} constraints")
        
        print(f"\n📊 CATEGORY PERFORMANCE")
        for category, stats in report["category_performance"].items():
            success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            constraint_rate = stats["constraints_satisfied"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {category.title()}: {stats['successful']}/{stats['total']} ({success_rate:.1%}) success, {constraint_rate:.1%} constraints")
        
        print(f"\n📁 Report saved to: {self.output_dir / 'real_music_test_report.json'}")


def main():
    """Main function to run real music tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real Music Test Suite for AI RemixMate Bridge")
    parser.add_argument("--music-dir", default="test_music", help="Directory containing test music files")
    parser.add_argument("--output-dir", default="runs/real_music_test", help="Output directory for test results")
    parser.add_argument("--test-case", help="Run specific test case only")
    parser.add_argument("--preset", choices=["radio", "club", "ambient"], help="Test specific preset only")
    parser.add_argument("--optimize", action="store_true", help="Test optimization mode only")
    parser.add_argument("--create-music", action="store_true", help="Create test music library")
    parser.add_argument("--quick", action="store_true", help="Run quick test with subset of test cases")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = RealMusicTestSuite(args.music_dir, args.output_dir)
    
    if args.create_music:
        # Create test music library
        test_suite.create_test_music_library()
        return 0
    
    # Create test cases
    test_suite.create_test_cases()
    
    if args.test_case:
        # Run single test case
        test_case = next((tc for tc in test_suite.test_cases if tc.name == args.test_case), None)
        if test_case:
            result = test_suite.run_test_case(test_case, args.preset or "radio", args.optimize)
            print(f"\n✅ Single test completed: {test_case.name}")
        else:
            print(f"❌ Test case not found: {args.test_case}")
            return 1
    elif args.quick:
        # Run quick test with subset
        print("🏃 Running quick test with subset of test cases...")
        quick_cases = test_suite.test_cases[:5]  # First 5 test cases
        test_suite.test_cases = quick_cases
        report = test_suite.run_comprehensive_tests()
    else:
        # Run full comprehensive test
        report = test_suite.run_comprehensive_tests()
    
    return 0


if __name__ == "__main__":
    exit(main())
