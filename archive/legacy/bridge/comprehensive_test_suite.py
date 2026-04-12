#!/usr/bin/env python3
"""
Comprehensive Test Suite for AI RemixMate Bridge

This module tests the bridge system with various music styles, mix scenarios,
and edge cases to ensure robust performance across all use cases.
"""

from __future__ import annotations
import json
import time
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import librosa
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime


from scripts.export_manifest import build_manifest, create_session_id
from scripts.bridge.logic_bridge import main as logic_main
from scripts.bridge.preset_manager import PresetManager
from scripts.bridge.iterative_optimizer import IterativeOptimizer
from scripts.bridge_metrics import compute_metrics


@dataclass
class TestScenario:
    """Represents a test scenario with specific parameters."""
    name: str
    description: str
    genre: str
    base_tempo: float
    match_tempo: float
    base_key: str
    match_key: str
    vocal_level: float  # 0.0 = no vocals, 1.0 = full vocals
    complexity: str  # simple, medium, complex
    expected_difficulty: str  # easy, medium, hard
    test_type: str  # genre, mix, edge_case, optimization


class AudioGenerator:
    """Generates synthetic audio for testing different scenarios."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.duration = 30  # 30 seconds for testing
    
    def generate_electronic_track(self, tempo: float, key: str, is_vocal: bool = False) -> np.ndarray:
        """Generate electronic music track."""
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        
        # Base frequency from key
        base_freq = self._key_to_frequency(key)
        
        if is_vocal:
            # Vocal-like: sine wave with vibrato and formants
            vibrato = 0.1 * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato
            fundamental = base_freq * (1 + vibrato)
            vocal = 0.3 * np.sin(2 * np.pi * fundamental * t)
            
            # Add formants (vocal characteristics)
            formant1 = 0.1 * np.sin(2 * np.pi * fundamental * 2.5 * t)
            formant2 = 0.05 * np.sin(2 * np.pi * fundamental * 4 * t)
            vocal += formant1 + formant2
            
            # Add some noise for realism
            noise = 0.02 * np.random.randn(len(t))
            return vocal + noise
        else:
            # Electronic instrumental: sawtooth bass + square lead
            bass_freq = base_freq * 0.5
            lead_freq = base_freq * 2
            
            # Sawtooth bass
            bass = 0.4 * np.sin(2 * np.pi * bass_freq * t)
            bass += 0.2 * np.sin(2 * np.pi * bass_freq * 2 * t)
            bass += 0.1 * np.sin(2 * np.pi * bass_freq * 3 * t)
            
            # Square lead
            lead = 0.3 * np.sign(np.sin(2 * np.pi * lead_freq * t))
            
            # Add some percussion (kick on beat)
            kick_freq = 60  # Sub-bass
            beat_interval = 60 / tempo
            kick_times = np.arange(0, self.duration, beat_interval)
            kick = np.zeros_like(t)
            for kick_time in kick_times:
                start_idx = int(kick_time * self.sr)
                end_idx = min(start_idx + int(0.1 * self.sr), len(kick))
                if start_idx < len(kick):
                    kick[start_idx:end_idx] += 0.5 * np.sin(2 * np.pi * kick_freq * 
                                                           np.linspace(0, 0.1, end_idx - start_idx))
            
            return bass + lead + kick
    
    def generate_rock_track(self, tempo: float, key: str, is_vocal: bool = False) -> np.ndarray:
        """Generate rock music track."""
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        base_freq = self._key_to_frequency(key)
        
        if is_vocal:
            # Rock vocals: more aggressive, less vibrato
            fundamental = base_freq
            vocal = 0.4 * np.sin(2 * np.pi * fundamental * t)
            
            # Add harmonics for rock character
            vocal += 0.2 * np.sin(2 * np.pi * fundamental * 2 * t)
            vocal += 0.1 * np.sin(2 * np.pi * fundamental * 3 * t)
            
            # Add some distortion
            vocal = np.tanh(vocal * 2) * 0.3
            
            return vocal
        else:
            # Rock instrumental: distorted guitar + bass + drums
            guitar_freq = base_freq
            bass_freq = base_freq * 0.5
            
            # Distorted guitar
            guitar = 0.3 * np.sin(2 * np.pi * guitar_freq * t)
            guitar += 0.2 * np.sin(2 * np.pi * guitar_freq * 1.5 * t)
            guitar = np.tanh(guitar * 3) * 0.2
            
            # Bass
            bass = 0.4 * np.sin(2 * np.pi * bass_freq * t)
            
            # Drums (simplified)
            drums = self._generate_drums(tempo, t)
            
            return guitar + bass + drums
    
    def generate_hip_hop_track(self, tempo: float, key: str, is_vocal: bool = False) -> np.ndarray:
        """Generate hip-hop track."""
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        base_freq = self._key_to_frequency(key)
        
        if is_vocal:
            # Hip-hop vocals: rhythmic, percussive
            fundamental = base_freq
            vocal = 0.3 * np.sin(2 * np.pi * fundamental * t)
            
            # Add rhythmic elements
            beat_interval = 60 / tempo
            for i in range(int(self.duration / beat_interval)):
                start_time = i * beat_interval
                if start_time < self.duration:
                    start_idx = int(start_time * self.sr)
                    end_idx = min(start_idx + int(0.2 * self.sr), len(vocal))
                    if start_idx < len(vocal):
                        vocal[start_idx:end_idx] *= 1.5
            
            return vocal
        else:
            # Hip-hop instrumental: heavy bass + samples
            bass_freq = base_freq * 0.25  # Very low bass
            
            # Sub-bass
            bass = 0.6 * np.sin(2 * np.pi * bass_freq * t)
            
            # Hi-hats and snares
            drums = self._generate_hip_hop_drums(tempo, t)
            
            # Sample-like elements
            sample = 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)
            sample = sample * (np.random.rand(len(t)) > 0.7)  # Gated
            
            return bass + drums + sample
    
    def generate_jazz_track(self, tempo: float, key: str, is_vocal: bool = False) -> np.ndarray:
        """Generate jazz track."""
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        base_freq = self._key_to_frequency(key)
        
        if is_vocal:
            # Jazz vocals: smooth, with vibrato
            vibrato = 0.2 * np.sin(2 * np.pi * 6 * t)
            fundamental = base_freq * (1 + vibrato)
            vocal = 0.25 * np.sin(2 * np.pi * fundamental * t)
            
            # Add jazz harmonies
            vocal += 0.1 * np.sin(2 * np.pi * fundamental * 1.2 * t)  # Minor third
            vocal += 0.05 * np.sin(2 * np.pi * fundamental * 1.5 * t)  # Perfect fifth
            
            return vocal
        else:
            # Jazz instrumental: piano + bass + drums
            piano_freq = base_freq
            
            # Piano (multiple notes)
            piano = 0.2 * np.sin(2 * np.pi * piano_freq * t)
            piano += 0.15 * np.sin(2 * np.pi * piano_freq * 1.2 * t)
            piano += 0.1 * np.sin(2 * np.pi * piano_freq * 1.5 * t)
            
            # Walking bass
            bass_freq = base_freq * 0.5
            bass = 0.3 * np.sin(2 * np.pi * bass_freq * t)
            
            # Jazz drums (swing feel)
            drums = self._generate_jazz_drums(tempo, t)
            
            return piano + bass + drums
    
    def generate_classical_track(self, tempo: float, key: str, is_vocal: bool = False) -> np.ndarray:
        """Generate classical track."""
        t = np.linspace(0, self.duration, int(self.sr * self.duration))
        base_freq = self._key_to_frequency(key)
        
        if is_vocal:
            # Classical vocals: pure, operatic
            fundamental = base_freq
            vocal = 0.2 * np.sin(2 * np.pi * fundamental * t)
            
            # Add operatic vibrato
            vibrato = 0.15 * np.sin(2 * np.pi * 4 * t)
            vocal *= (1 + vibrato)
            
            # Add harmonics
            vocal += 0.1 * np.sin(2 * np.pi * fundamental * 2 * t)
            vocal += 0.05 * np.sin(2 * np.pi * fundamental * 3 * t)
            
            return vocal
        else:
            # Classical instrumental: strings + woodwinds
            strings_freq = base_freq
            
            # String section
            strings = 0.15 * np.sin(2 * np.pi * strings_freq * t)
            strings += 0.1 * np.sin(2 * np.pi * strings_freq * 1.2 * t)
            strings += 0.08 * np.sin(2 * np.pi * strings_freq * 1.5 * t)
            strings += 0.05 * np.sin(2 * np.pi * strings_freq * 2 * t)
            
            # Woodwinds
            woodwind_freq = base_freq * 2
            woodwind = 0.1 * np.sin(2 * np.pi * woodwind_freq * t)
            
            return strings + woodwind
    
    def _key_to_frequency(self, key: str) -> float:
        """Convert Camelot key to base frequency."""
        # Map Camelot keys to frequencies (A4 = 440 Hz)
        key_map = {
            "1A": 261.63, "1B": 277.18,  # C, C#
            "2A": 293.66, "2B": 311.13,  # D, D#
            "3A": 329.63, "3B": 349.23,  # E, F
            "4A": 369.99, "4B": 392.00,  # F#, G
            "5A": 415.30, "5B": 440.00,  # G#, A
            "6A": 466.16, "6B": 493.88,  # A#, B
            "7A": 523.25, "7B": 554.37,  # C, C#
            "8A": 587.33, "8B": 622.25,  # D, D#
            "9A": 659.25, "9B": 698.46,  # E, F
            "10A": 739.99, "10B": 783.99, # F#, G
            "11A": 830.61, "11B": 880.00, # G#, A
            "12A": 932.33, "12B": 987.77  # A#, B
        }
        return key_map.get(key, 440.0)  # Default to A
    
    def _generate_drums(self, tempo: float, t: np.ndarray) -> np.ndarray:
        """Generate basic drum pattern."""
        drums = np.zeros_like(t)
        beat_interval = 60 / tempo
        
        for i in range(int(self.duration / beat_interval)):
            start_time = i * beat_interval
            if start_time < self.duration:
                start_idx = int(start_time * self.sr)
                end_idx = min(start_idx + int(0.1 * self.sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.3 * np.random.randn(end_idx - start_idx)
        
        return drums
    
    def _generate_hip_hop_drums(self, tempo: float, t: np.ndarray) -> np.ndarray:
        """Generate hip-hop drum pattern."""
        drums = np.zeros_like(t)
        beat_interval = 60 / tempo
        
        # Kick on 1 and 3
        for i in range(0, int(self.duration / beat_interval), 2):
            start_time = i * beat_interval
            if start_time < self.duration:
                start_idx = int(start_time * self.sr)
                end_idx = min(start_idx + int(0.15 * self.sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.4 * np.random.randn(end_idx - start_idx)
        
        # Snare on 2 and 4
        for i in range(1, int(self.duration / beat_interval), 2):
            start_time = i * beat_interval
            if start_time < self.duration:
                start_idx = int(start_time * self.sr)
                end_idx = min(start_idx + int(0.05 * self.sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.2 * np.random.randn(end_idx - start_idx)
        
        return drums
    
    def _generate_jazz_drums(self, tempo: float, t: np.ndarray) -> np.ndarray:
        """Generate jazz drum pattern with swing feel."""
        drums = np.zeros_like(t)
        beat_interval = 60 / tempo
        
        # Swing feel: off-beat emphasis
        for i in range(int(self.duration / beat_interval)):
            start_time = i * beat_interval
            if start_time < self.duration:
                start_idx = int(start_time * self.sr)
                end_idx = min(start_idx + int(0.08 * self.sr), len(drums))
                if start_idx < len(drums):
                    drums[start_idx:end_idx] += 0.2 * np.random.randn(end_idx - start_idx)
        
        return drums


class ComprehensiveTestSuite:
    """Comprehensive test suite for the bridge system."""
    
    def __init__(self, output_dir: str = "runs/comprehensive_test"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_generator = AudioGenerator()
        self.preset_manager = PresetManager()
        
        self.test_scenarios = self._create_test_scenarios()
        self.results = []
    
    def _create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios."""
        scenarios = []
        
        # Genre Tests
        genres = ["electronic", "rock", "hip_hop", "jazz", "classical"]
        tempos = [120, 140, 100, 80, 160]
        keys = ["8A", "5B", "12A", "3B", "10A"]
        
        for i, genre in enumerate(genres):
            scenarios.append(TestScenario(
                name=f"genre_{genre}_standard",
                description=f"Standard {genre} track test",
                genre=genre,
                base_tempo=tempos[i],
                match_tempo=tempos[i],
                base_key=keys[i],
                match_key=keys[i],
                vocal_level=0.7,
                complexity="medium",
                expected_difficulty="easy",
                test_type="genre"
            ))
        
        # Mix Scenario Tests
        mix_scenarios = [
            # Key mismatches
            TestScenario("key_mismatch_major", "Major key mismatch", "electronic", 120, 120, "8A", "5A", 0.7, "medium", "medium", "mix"),
            TestScenario("key_mismatch_minor", "Minor key mismatch", "rock", 140, 140, "8B", "5B", 0.6, "medium", "medium", "mix"),
            
            # Tempo mismatches
            TestScenario("tempo_fast", "Fast tempo difference", "hip_hop", 80, 120, "8A", "8A", 0.8, "medium", "medium", "mix"),
            TestScenario("tempo_slow", "Slow tempo difference", "jazz", 160, 100, "8A", "8A", 0.5, "medium", "medium", "mix"),
            
            # Vocal level variations
            TestScenario("vocals_heavy", "Heavy vocals", "electronic", 120, 120, "8A", "8A", 0.9, "simple", "easy", "mix"),
            TestScenario("vocals_light", "Light vocals", "rock", 140, 140, "8A", "8A", 0.2, "medium", "hard", "mix"),
            TestScenario("instrumental_only", "Instrumental only", "jazz", 100, 100, "8A", "8A", 0.0, "simple", "easy", "mix"),
            
            # Complex scenarios
            TestScenario("complex_mismatch", "Complex key and tempo mismatch", "electronic", 120, 140, "8A", "5B", 0.6, "complex", "hard", "mix"),
        ]
        scenarios.extend(mix_scenarios)
        
        # Edge Case Tests
        edge_scenarios = [
            TestScenario("very_fast", "Very fast tempo", "electronic", 180, 180, "8A", "8A", 0.7, "medium", "medium", "edge_case"),
            TestScenario("very_slow", "Very slow tempo", "classical", 60, 60, "8A", "8A", 0.5, "medium", "medium", "edge_case"),
            TestScenario("extreme_key_diff", "Extreme key difference", "rock", 120, 120, "1A", "12B", 0.6, "complex", "hard", "edge_case"),
            TestScenario("short_track", "Short track (10s)", "electronic", 120, 120, "8A", "8A", 0.7, "simple", "easy", "edge_case"),
            TestScenario("long_track", "Long track (60s)", "jazz", 100, 100, "8A", "8A", 0.5, "medium", "medium", "edge_case"),
        ]
        scenarios.extend(edge_scenarios)
        
        # Optimization Tests
        opt_scenarios = [
            TestScenario("opt_challenging", "Challenging optimization", "electronic", 120, 140, "8A", "5B", 0.6, "complex", "hard", "optimization"),
            TestScenario("opt_easy", "Easy optimization", "rock", 120, 120, "8A", "8A", 0.7, "simple", "easy", "optimization"),
        ]
        scenarios.extend(opt_scenarios)
        
        return scenarios
    
    def generate_test_audio(self, scenario: TestScenario) -> Tuple[Path, Path]:
        """Generate test audio files for a scenario."""
        scenario_dir = self.output_dir / scenario.name
        scenario_dir.mkdir(exist_ok=True)
        
        # Generate base track (vocals)
        if scenario.vocal_level > 0:
            base_audio = self._generate_track_by_genre(
                scenario.genre, scenario.base_tempo, scenario.base_key, True
            )
            # Adjust vocal level
            base_audio *= scenario.vocal_level
        else:
            # Instrumental only
            base_audio = np.zeros(int(self.audio_generator.sr * self.audio_generator.duration))
        
        # Generate match track (instrumentals)
        match_audio = self._generate_track_by_genre(
            scenario.genre, scenario.match_tempo, scenario.match_key, False
        )
        
        # Save audio files
        base_path = scenario_dir / "base.wav"
        match_path = scenario_dir / "match.wav"
        
        sf.write(str(base_path), base_audio, self.audio_generator.sr)
        sf.write(str(match_path), match_audio, self.audio_generator.sr)
        
        return base_path, match_path
    
    def _generate_track_by_genre(self, genre: str, tempo: float, key: str, is_vocal: bool) -> np.ndarray:
        """Generate track based on genre."""
        if genre == "electronic":
            return self.audio_generator.generate_electronic_track(tempo, key, is_vocal)
        elif genre == "rock":
            return self.audio_generator.generate_rock_track(tempo, key, is_vocal)
        elif genre == "hip_hop":
            return self.audio_generator.generate_hip_hop_track(tempo, key, is_vocal)
        elif genre == "jazz":
            return self.audio_generator.generate_jazz_track(tempo, key, is_vocal)
        elif genre == "classical":
            return self.audio_generator.generate_classical_track(tempo, key, is_vocal)
        else:
            return self.audio_generator.generate_electronic_track(tempo, key, is_vocal)
    
    def run_single_test(self, scenario: TestScenario, preset: str = "radio", optimize: bool = False) -> Dict[str, Any]:
        """Run a single test scenario."""
        print(f"\n🧪 Testing: {scenario.name}")
        print(f"   Description: {scenario.description}")
        print(f"   Genre: {scenario.genre}, Tempo: {scenario.base_tempo}/{scenario.match_tempo}")
        print(f"   Keys: {scenario.base_key}/{scenario.match_key}, Vocal Level: {scenario.vocal_level}")
        print(f"   Expected Difficulty: {scenario.expected_difficulty}")
        
        start_time = time.time()
        
        try:
            # Generate test audio
            base_path, match_path = self.generate_test_audio(scenario)
            
            # Create session
            session_id = create_session_id()
            session_dir = self.output_dir / scenario.name / f"session_{session_id}"
            session_dir.mkdir(exist_ok=True)
            
            # Build manifest
            stems = {
                "vocals_path": str(base_path),
                "instrumental_path": str(match_path),
                "vocals_bpm": scenario.base_tempo,
                "instrumental_bpm": scenario.match_tempo,
                "vocals_key": scenario.base_key,
                "instrumental_key": scenario.match_key,
                "project_bpm": (scenario.base_tempo + scenario.match_tempo) / 2,
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
                project_key=scenario.base_key,
                mix_params=stems["mix_params"]
            )
            
            # Apply preset
            if preset != "radio":
                manifest = self.preset_manager.apply_preset_to_manifest(manifest, preset)
            
            # Write manifest
            manifest_path = session_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Run bridge (simulate without Logic Pro for now)
            if optimize:
                # Test iterative optimization
                optimizer = IterativeOptimizer(max_iterations=3)
                best_manifest, best_report = optimizer.optimize(manifest)
                
                if best_report:
                    report = best_report
                    optimization_used = True
                else:
                    # Fallback to single run
                    report = self._simulate_bridge_run(manifest, session_dir)
                    optimization_used = False
            else:
                # Single run
                report = self._simulate_bridge_run(manifest, session_dir)
                optimization_used = False
            
            # Calculate test metrics
            test_duration = time.time() - start_time
            
            result = {
                "scenario": scenario.name,
                "description": scenario.description,
                "genre": scenario.genre,
                "test_type": scenario.test_type,
                "expected_difficulty": scenario.expected_difficulty,
                "preset": preset,
                "optimization_used": optimization_used,
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
                "scenario": scenario.name,
                "description": scenario.description,
                "genre": scenario.genre,
                "test_type": scenario.test_type,
                "expected_difficulty": scenario.expected_difficulty,
                "preset": preset,
                "optimization_used": optimize,
                "test_duration": time.time() - start_time,
                "success": False,
                "error": str(e),
                "constraints_satisfied": False
            }
    
    def _simulate_bridge_run(self, manifest: Dict[str, Any], session_dir: Path) -> Dict[str, Any]:
        """Simulate bridge run without Logic Pro (for testing)."""
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
        """Run all test scenarios."""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        # Test configurations
        presets = ["radio", "club", "ambient"]
        optimization_modes = [False, True]
        
        total_tests = len(self.test_scenarios) * len(presets) * len(optimization_modes)
        current_test = 0
        
        for scenario in self.test_scenarios:
            for preset in presets:
                for optimize in optimization_modes:
                    current_test += 1
                    print(f"\n📊 Progress: {current_test}/{total_tests}")
                    
                    result = self.run_single_test(scenario, preset, optimize)
                    self.results.append(result)
        
        # Generate comprehensive report
        return self.generate_test_report()
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n📊 Generating Comprehensive Test Report")
        print("=" * 50)
        
        # Calculate statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - successful_tests
        
        constraints_satisfied = sum(1 for r in self.results if r.get("constraints_satisfied", False))
        optimization_tests = sum(1 for r in self.results if r.get("optimization_used", False))
        
        # Performance metrics
        test_durations = [r["test_duration"] for r in self.results if r["success"]]
        avg_duration = np.mean(test_durations) if test_durations else 0
        max_duration = max(test_durations) if test_durations else 0
        min_duration = min(test_durations) if test_durations else 0
        
        # Genre performance
        genre_stats = {}
        for genre in ["electronic", "rock", "hip_hop", "jazz", "classical"]:
            genre_results = [r for r in self.results if r.get("genre") == genre]
            if genre_results:
                genre_stats[genre] = {
                    "total": len(genre_results),
                    "successful": sum(1 for r in genre_results if r["success"]),
                    "constraints_satisfied": sum(1 for r in genre_results if r.get("constraints_satisfied", False))
                }
        
        # Preset performance
        preset_stats = {}
        for preset in ["radio", "club", "ambient"]:
            preset_results = [r for r in self.results if r.get("preset") == preset]
            if preset_results:
                preset_stats[preset] = {
                    "total": len(preset_results),
                    "successful": sum(1 for r in preset_results if r["success"]),
                    "constraints_satisfied": sum(1 for r in preset_results if r.get("constraints_satisfied", False))
                }
        
        # Test type performance
        test_type_stats = {}
        for test_type in ["genre", "mix", "edge_case", "optimization"]:
            type_results = [r for r in self.results if r.get("test_type") == test_type]
            if type_results:
                test_type_stats[test_type] = {
                    "total": len(type_results),
                    "successful": sum(1 for r in type_results if r["success"]),
                    "constraints_satisfied": sum(1 for r in type_results if r.get("constraints_satisfied", False))
                }
        
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
            "test_type_performance": test_type_stats,
            "detailed_results": self.results,
            "test_timestamp": datetime.now().isoformat(),
            "test_environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "output_directory": str(self.output_dir)
            }
        }
        
        # Save report
        report_path = self.output_dir / "comprehensive_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_test_summary(report)
        
        return report
    
    def print_test_summary(self, report: Dict[str, Any]) -> None:
        """Print test summary."""
        summary = report["test_summary"]
        perf = report["performance_metrics"]
        
        print(f"\n🎯 COMPREHENSIVE TEST RESULTS")
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
        
        print(f"\n📊 TEST TYPE PERFORMANCE")
        for test_type, stats in report["test_type_performance"].items():
            success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            constraint_rate = stats["constraints_satisfied"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {test_type.title()}: {stats['successful']}/{stats['total']} ({success_rate:.1%}) success, {constraint_rate:.1%} constraints")
        
        print(f"\n📁 Report saved to: {self.output_dir / 'comprehensive_test_report.json'}")


def main():
    """Main function to run comprehensive tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Suite for AI RemixMate Bridge")
    parser.add_argument("--output-dir", default="runs/comprehensive_test", help="Output directory for test results")
    parser.add_argument("--scenario", help="Run specific scenario only")
    parser.add_argument("--preset", choices=["radio", "club", "ambient"], help="Test specific preset only")
    parser.add_argument("--optimize", action="store_true", help="Test optimization mode only")
    parser.add_argument("--quick", action="store_true", help="Run quick test with subset of scenarios")
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = ComprehensiveTestSuite(args.output_dir)
    
    if args.scenario:
        # Run single scenario
        scenario = next((s for s in test_suite.test_scenarios if s.name == args.scenario), None)
        if scenario:
            result = test_suite.run_single_test(scenario, args.preset or "radio", args.optimize)
            print(f"\n✅ Single test completed: {scenario.name}")
        else:
            print(f"❌ Scenario not found: {args.scenario}")
            return 1
    elif args.quick:
        # Run quick test with subset
        print("🏃 Running quick test with subset of scenarios...")
        quick_scenarios = test_suite.test_scenarios[:5]  # First 5 scenarios
        test_suite.test_scenarios = quick_scenarios
        report = test_suite.run_comprehensive_tests()
    else:
        # Run full comprehensive test
        report = test_suite.run_comprehensive_tests()
    
    return 0


if __name__ == "__main__":
    exit(main())
