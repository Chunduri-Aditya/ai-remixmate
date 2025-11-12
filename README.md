# 🎵 AI RemixMate – Arrangement-Level AI DJ Mixer

> **TL;DR**: Upload 2 songs → AI splits stems, detects structure, plans a DJ-style arrangement, and renders a beatmatched, key-compatible mix with pro FX. Use Classic (fast), Intelligent (auto), or Arrangement (timeline) modes via web UI or Python API.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Chunduri-Aditya/ai-remixmate.git
cd ai-remixmate

pip install -r requirements.txt
python app.py
```

**That's it!** The app launches on `http://127.0.0.1:7860` with automatic dependency checking.

---

## ✨ What It Does

**AI RemixMate** transforms two audio tracks into a professional DJ mix by:

- **Separating stems** (vocals, drums, bass, other) using Demucs
- **Detecting structure** (intro, build, drop, break, outro) via spectral analysis
- **Planning arrangements** using genre-aware templates (EDM, Hip-hop, Acapella, Progressive)
- **Rendering timeline-based mixes** with beatmatching, key-matching, and effects
- **Learning from preferences** to optimize mixing strategies

**Three Mixing Modes:**

1. **Classic**: Simple stem swaps + crossfades
2. **Intelligent**: AI chooses best DJ technique (harmonic mixing, dynamic curves)
3. **Arrangement**: Structure-aware, timeline-based DJ set ⭐

---

## 🎯 Key Features

### ✅ Fully Implemented

- ✅ **Multi-format support** (MP3, M4A, FLAC, OGG, AAC, WMA, AIFF, WAV) with no duration limits
- ✅ **Real-time track analysis** (BPM, key, genre, energy, danceability)
- ✅ **Song recommendations** based on compatibility (BPM, key, genre, features)
- ✅ **Intelligent volume balancing** (vocal boost 50%, instrumental reduction 30%)
- ✅ **Dynamic crossfade curves** (linear, log, exp, s-curve, cosine, sigmoid)
- ✅ **Harmonic mixing** (Camelot Wheel compatibility, key-shifting)
- ✅ **Playlist management** (create, manage, view playlists)
- ✅ **Auto mode selection** (automatically picks best remix mode)
- ✅ **Structure detection** (beat grid, bars, phrases, sections)
- ✅ **Template-based planning** (EDM, Hip-hop, Acapella, Progressive)
- ✅ **Energy curve planning** (Chill→Peak, Rollercoaster, Slow Build, Double Peak)
- ✅ **Phrase alignment** (8/16-bar boundary enforcement)

### 🟡 Partially Implemented

- 🟡 **ML Models**: Basic models with rule-based fallbacks (needs training data)
- 🟡 **Advanced Arrangement**: Sidechain ducking, bus processing (basic implementations)
- 🟡 **LUFS Normalization**: Currently uses peak normalization only

### 🔴 Planned

- 🔴 **Caching System**: Cache stems/analysis by file hash
- 🔴 **LUFS Normalization**: Full EBU R128 implementation
- 🔴 **True-Peak Limiter**: Advanced limiter with inter-sample peak detection
- 🔴 **FastAPI REST API**: Currently Gradio only
- 🔴 **GPU Acceleration**: Optional GPU for Demucs

---

## 🏗️ Architecture

### Complete Pipeline

```
User Uploads 2 Tracks (Any Format)
        ↓
[1] Audio Conversion → WAV (librosa/pydub)
        ↓
[2] Stem Separation → Vocals, Drums, Bass, Other (Demucs)
        ↓
[3] Track Analysis → BPM, Key, Genre, Energy, Danceability
        ↓
[4] Structure Detection → Beats → Bars → Phrases → Sections
        ↓
[5] Mixing Mode Selection:
    • Classic: Stem swaps + crossfades
    • Intelligent: AI-optimized technique
    • Arrangement: Structure-aware timeline ⭐
        ↓
[6] Arrangement Planning (if Arrangement mode):
    • Template selection (EDM/Hip-hop/Acapella/Progressive)
    • Energy curve planning
    • Phrase-aligned transitions
        ↓
[7] Timeline Rendering:
    • Time-stretch for beatmatching
    • Apply effects (EQ, reverb, delay)
    • Mix clips with crossfades
        ↓
[8] Post-Processing:
    • Volume balancing (vocal boost, instrumental reduction)
    • Normalization (peak + LUFS planned)
        ↓
Final Professional DJ Mix (WAV)
```

### Example: Two Songs to Mix

**Input:**
- Track 1: EDM, 128 BPM, key 8A, 3:45
- Track 2: Pop, 130 BPM, key 9A, 3:20

**Process:**

| Timestamp | What Happens |
|-----------|--------------|
| **0:00-0:16** | Intro(A) - Clean start, low energy (0.3) |
| **0:16-0:48** | Build(A) - Energy rising (0.3→0.7), overlay drums(B) |
| **0:48-1:36** | Drop(A) - Peak energy (0.9), full arrangement |
| **1:36-1:52** | **Transition** - Bass swap, overlay vocals(B) on Drop(A) |
| **1:52-2:40** | Drop(B) + Vocals(A) - Mashup moment, energy maintained |
| **2:40-3:20** | Outro(B) - Fade out, energy decreasing |

**Output:** 3:20 professional mix, beatmatched (128 BPM), key-compatible (8A↔9A), structure-aware.

---

## 📁 Project Structure

```
.
├── app.py                          # Gradio web application
├── requirements.txt                # Dependencies
├── sitecustomize.py                # Path configuration
├── test_project.py                 # Test suite
│
├── src/remixmate/                  # Core package
│   ├── config.py                   # Configuration
│   ├── remix_core.py               # Main orchestrator
│   ├── dj_mixing.py                # DJ techniques
│   ├── structure_detection.py     # Structure analysis
│   ├── timeline_planner.py        # Arrangement planning
│   ├── timeline_renderer.py       # Timeline rendering
│   ├── recommendations.py          # Analysis & recommendations
│   ├── ml_audio_features.py        # ML models
│   ├── auto_mode_selector.py       # Auto mode selection
│   ├── advanced_arrangement.py     # Advanced features
│   ├── lyrics_extraction.py        # Lyrics extraction
│   └── playlist_manager.py         # Playlist management
│
├── scripts/                        # Utility scripts
│   ├── ingestion/                  # Audio ingestion
│   ├── database/                   # Database operations
│   ├── audio_processing/          # Audio processing
│   ├── analysis/                   # Analysis scripts
│   └── remixing/                   # Remix generation
│
├── audio_input/                    # User uploads (gitignored)
├── audio_output/                   # Generated remixes (gitignored)
├── models/                         # Database & ML models (gitignored)
└── [other directories]            # User data (gitignored)
```

---

## ⚙️ Configuration

### Core Settings

| Setting | Default | Range | Purpose |
|---------|---------|-------|---------|
| `SAMPLE_RATE` | 44100 Hz | Fixed | Audio processing sample rate |
| `MAX_DURATION_SEC` | None | Unlimited | No duration limits |
| `max_bpm_stretch_ratio` | 1.15 | 1.00-1.25 | Limits time-stretch artifacts |
| `vocal_boost` | 1.5x | 1.0-2.0 | Vocal presence in mix |
| `instrumental_reduction` | 0.7x | 0.5-1.0 | Prevents instrumental masking |
| `crossfade_seconds` | 8.0 | 3-16 | Default transition length |
| `limiter_ceiling_dbTP` | -1.0 | -2.0 to -0.5 | True-peak safety margin |
| `target_loudness_LUFS-I` | -14 | -18 to -10 | Platform-consistent loudness (🔴 planned) |

### Energy Curve Presets

| Preset | Energy Array | Description |
|--------|--------------|-------------|
| `chill_to_peak` | [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] | Gradual increase |
| `rollercoaster` | [0.5, 0.8, 0.4, 0.9, 0.3, 0.8, 0.5, 0.7] | Multiple peaks |
| `slow_build` | [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] | Very gradual |
| `double_peak` | [0.4, 0.6, 0.8, 0.9, 0.7, 0.9, 0.8, 0.6] | Two energy peaks |

---

## 🛠️ Usage

### Web Interface

1. **Launch**: `python app.py`
2. **Upload** two audio files (any format)
3. **View** real-time analysis (BPM, key, genre, energy, danceability)
4. **Select** mode:
   - 🎯 **Auto (Recommended)**: Automatically selects best mode
   - **Mashup**: Mix both tracks together
   - **Track 1 Vocals + Track 2 Instruments**: Vocals from track 1, instruments from track 2
   - **Track 1 Instruments + Track 2 Vocals**: Instruments from track 1, vocals from track 2
5. **Enable**:
   - ✅ **Intelligent Mixing**: AI chooses best DJ technique
   - ✅ **Arrangement Mixing**: Structure-aware, timeline-based mixing
6. **Adjust** (when Arrangement Mixing enabled):
   - **Remix Aggressiveness** (0.0-1.0):
     - 0.0: Mostly keeps original structure, 1 big transition
     - 0.5: More creative re-usage of sections, one mashup moment
     - 1.0: Heavy slicing, multiple "wow moments"
   - **Energy Shape**: Chill→Peak, Rollercoaster, Slow Build, Double Peak
7. **Generate** remix and download

### Python API

#### Basic Remix
```python
from remixmate import remix_core

output = remix_core.remix_two_files(
    "track1.mp3",
    "track2.mp3",
    mode="mashup"
)
```

#### Arrangement Mixing
```python
output = remix_core.remix_two_files(
    "track1.mp3",
    "track2.mp3",
    mode="mashup",
    use_arrangement_mixing=True,
    aggressiveness=0.7,
    energy_shape="chill_to_peak"
)
```

#### Intelligent Mixing
```python
output = remix_core.remix_two_files(
    "track1.mp3",
    "track2.mp3",
    mode="mashup",
    use_intelligent_mixing=True,
    mixing_technique="crossfade",
    crossfade_length=8.0,
    apply_beatmatching=True
)
```

#### Track Analysis
```python
from remixmate import recommendations

# Analyze track
features = recommendations.analyze_track_characteristics("track.mp3")
print(f"BPM: {features['bpm']}, Key: {features['key']}, Genre: {features['genre']}")

# Get recommendations
recs = recommendations.find_compatible_songs("track.mp3", top_k=5)
for song_name, score, reasons in recs:
    print(f"{song_name}: {score*100:.0f}% match")
```

---

## 📊 Performance Benchmarks

### Test Setup
- **Hardware**: MacBook Pro (M1), 16GB RAM
- **Tracks**: 2 × 3-minute songs (MP3, 128kbps)
- **Format**: WAV output, 44.1kHz, stereo

### Stage Timings

| Stage | Time | Notes |
|-------|------|-------|
| Audio Conversion | 2-3s | librosa (fast) |
| Stem Separation | 45-60s | Demucs (CPU-intensive) |
| Track Analysis | 3-5s | First 60 seconds |
| Structure Detection | 8-12s | Beat grid + sections |
| Arrangement Planning | 1-2s | Template selection + planning |
| Timeline Rendering | 15-20s | Time-stretch + effects + mixing |
| Post-Processing | 2-3s | Volume balancing + normalization |
| **Total** | **76-105s** | ~1.5-2 minutes |

### Memory Usage
- **Peak RAM**: ~2-4GB (during stem separation)
- **Disk**: ~500MB per remix (stems + output)

---

## ⚠️ Limitations & Assumptions

### Stem Separation
- **Quality**: Depends on Demucs model (best with trained models)
- **Fallback**: Uses original audio if Demucs unavailable
- **Performance**: CPU/GPU intensive, scales with track length

### Structure Detection
- **4/4 Bias**: Works best on 4/4, steady-beat music (EDM, Pop, Hip-hop)
- **Time Signatures**: May struggle with complex time signatures (3/4, 7/8, etc.)
- **Probabilistic**: Section detection is probabilistic (may miss subtle transitions)
- **Minimum Length**: Requires 5+ seconds for reliable detection

### Beatmatching
- **BPM Limits**: Extreme differences (>20 BPM) may fall back to quick-cut
- **Artifacts**: Time-stretching artifacts possible with large stretches (>15%)
- **Grid Confidence**: Low confidence → conservative transitions

### ML Models
- **Training Data**: Genre classifier trained on limited dataset (10 genres)
- **Fallbacks**: Energy/danceability use heuristics when ML unavailable
- **Confidence**: Low confidence → conservative decisions (no key-shift, prefer quick-cut)

### Audio Formats
- **Conversion**: All formats converted to WAV internally (temporary disk usage)
- **Memory**: Very large files may require significant RAM
- **Processing**: No streaming support (loads full file into memory)

---

## 🔧 Installation & Dependencies

### Prerequisites
- Python 3.8+
- pip package manager

### Core Dependencies
```bash
pip install -r requirements.txt
```

**Core Libraries**:
- `numpy` (<2.0.0 for compatibility)
- `librosa` (audio analysis)
- `soundfile` (audio I/O)
- `scipy` (signal processing)
- `scikit-learn` (ML models)
- `gradio` (web interface)

**Optional (for stem separation)**:
- `torch` / `torchaudio` (PyTorch)
- `demucs` (stem separation)

**Optional (for lyrics)**:
- `openai-whisper` (best accuracy)
- `SpeechRecognition` (Google API fallback)
- `pocketsphinx` (offline fallback)

**Utilities**:
- `yt-dlp` (YouTube extraction)
- `pydub` (audio conversion fallback)

### Automatic Dependency Installation

The `app.py` script automatically:
1. Checks for missing dependencies
2. Installs them if missing
3. Verifies NumPy version compatibility
4. Launches the web application

---

## 🧪 Testing

### Quick Test
```bash
python test_project.py
```

### Individual Tests
```bash
# Test imports
python -c "from remixmate import config; print('✅ Imports work!')"

# Test dependencies
python -c "import numpy, librosa, soundfile; print('✅ Dependencies installed!')"

# Test NumPy compatibility
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
```

### Expected Output
```
🧪 AI RemixMate Project Tests
============================================================
✅ Passed: 9/9
🎉 All tests passed! Project is ready to use.
```

---

## 🔒 Security & Privacy

### Data Handling
- ✅ **Local Processing**: All audio processing happens locally (no cloud uploads)
- ✅ **No Telemetry**: No usage tracking or analytics
- ✅ **Cache Paths**: Stems and analysis cached locally in `separated/` and `models/`
- ✅ **Temporary Files**: Converted audio stored in `temp/converted/` (can be cleared)

### Copyright & Legal
- ⚠️ **Input Files**: User is responsible for copyright compliance
- ⚠️ **Remix Distribution**: User is responsible for licensing remixes
- ✅ **Model Licenses**: Demucs (MIT), Whisper (MIT), all clearly marked as optional

---

## 🛠️ Troubleshooting

### Common Issues

#### "Demucs not available"
**Impact**: Stem separation limited (uses original audio for all stems)  
**Solution**: Install Demucs for full functionality:
```bash
pip install demucs
```

#### "NumPy version incompatible"
**Impact**: May cause import errors  
**Solution**:
```bash
pip install "numpy>=1.24.0,<2.0.0"
```

#### "Audio too short for analysis"
**Impact**: Uses fallback sections  
**Solution**: Use tracks ≥5 seconds for reliable structure detection

#### "ModuleNotFoundError: No module named 'remixmate'"
**Impact**: Import errors  
**Solution**: Run from project root directory (path setup is automatic)

#### "Stem separation failed"
**Impact**: Falls back to original audio  
**Solution**: Check Demucs installation, disk space, file integrity

### Performance Tips
1. **Use WAV files** when possible (avoids conversion)
2. **Shorter analysis** (first 60 seconds) for faster recommendations
3. **Close other apps** during stem separation (CPU/GPU intensive)
4. **Enable caching** (🔴 planned) for repeated remixes

---

## 🎓 Key Concepts

### Camelot Wheel
Circular system for harmonic mixing:
- 12 numbers (1-12) = musical keys
- 2 letters (A/B) = major/minor
- Compatible keys are adjacent or same number

### Spectral Novelty
Measure of audio spectrum change over time:
- High novelty = significant change (section boundary)
- Low novelty = stable content (middle of section)

### Energy Curve
Target energy profile for DJ set:
- **Chill→Peak**: Gradual increase (0.3 → 1.0)
- **Rollercoaster**: Multiple peaks and valleys
- **Slow Build**: Very gradual increase
- **Double Peak**: Two energy peaks

### Phrase Alignment
Musical phrases are typically 8 or 16 bars:
- Transitions occur at phrase boundaries
- Prevents "mid-sentence" cuts
- Creates natural-sounding mixes

---

## 🔬 Technical Details

### Audio Conversion
- **Primary**: librosa (fast, high quality)
- **Fallback**: pydub (if librosa fails)
- **Formats**: MP3, M4A, FLAC, OGG, AAC, WMA, AIFF, WAV
- **Output**: WAV, 44.1kHz, stereo

### Stem Separation
- **Method**: Demucs (PyTorch-based)
- **Stems**: Vocals, Drums, Bass, Other
- **Fallback**: Original audio if Demucs unavailable

### Structure Detection
- **Beat Grid**: librosa beat tracking → beats → bars (4 beats) → phrases (8 bars)
- **Sections**: Spectral novelty + energy analysis + vocal detection
- **Labels**: Intro, Verse, Build, Drop, Break, Outro

### Arrangement Planning
- **Templates**: EDM Standard, Hip-hop Cut, Acapella Blend, Progressive Build
- **Energy Matching**: Select sections matching target energy curve
- **Phrase Alignment**: Snap transitions to 8/16-bar boundaries

### Volume Balancing
- **Process**: Calculate RMS → normalize to average → boost vocals 50% → reduce instrumentals 30%
- **Prevent Clipping**: Normalize to peak if >1.0

### Harmonic Mixing
- **Compatibility**: Same number, adjacent numbers, opposite letter
- **Key Shifting**: Pitch-shift audio to match incompatible keys

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details.

**Model Licenses**:
- Demucs: MIT
- Whisper: MIT
- All clearly marked as optional dependencies

---

## 🤝 Contributing

Contributions welcome! The project uses:
- Modular architecture (easy to extend)
- Clear separation of concerns
- Comprehensive error handling
- Graceful degradation (fallbacks for optional features)

---

## 📚 Additional Resources

### First Time Setup
On first run, the app will:
- Download Demucs models (~1GB) - takes 5-10 minutes
- Create necessary directories
- Set up the database structure

### Notes
- App runs on port 7860 by default
- All audio files processed in background
- Remixes saved to `audio_output/` directory
- Playlists stored in `models/user_playlists.json`

---

*Last Updated: 2024*
