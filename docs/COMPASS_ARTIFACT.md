# Music Tokenization for AI/ML: Complete Technical Reference

**Every major approach to converting music into mathematical representations for neural networks, organized for the AI RemixMate project.** This document covers symbolic, spectral, neural, harmonic, and multimodal tokenization methods with exact math, computational costs, tradeoffs, code references, and integration strategies. The field spans three paradigms: classical signal processing (STFT, mel, CQT), symbolic encoding (MIDI tokenization schemes), and learned discrete representations (VQ-VAE, RVQ codecs). For AI RemixMate's stack — Demucs separation, Krumhansl-Schmuckler key detection, LUFS mastering, Camelot Wheel mixing — the most actionable path combines **CQT chroma for harmonic analysis**, **EnCodec/DAC tokens for generative manipulation**, and **VampNet for creative inpainting**, all orchestrated through the existing FastAPI backend.

---

## 1. Symbolic tokenization: MIDI and score representations

### 1.1 MIDI tokenization strategies

MIDI encodes performance data as events: pitch (0–127), velocity (0–127), timing (in ticks), and channel/program (0–127). Converting these events into discrete token sequences for transformer models requires strategic design choices about how to represent notes, time, and structure. Six major strategies dominate the field.

**MIDILike** (Oore et al., 2018; Music Transformer, Huang et al., 2018) uses raw event tokens: `NoteOn_{pitch}` (128 tokens), `NoteOff_{pitch}` (128 tokens), `Velocity_{v}` (32 bins, bin = floor(velocity/4)), and `TimeShift_{t}` (100 tokens at 10ms resolution, representing 10ms–1000ms shifts). A note at 523ms becomes `TimeShift_52`. Total vocabulary: **~388–500 tokens**. Average sequence length: ~1,772 tokens per piece. Preserves expressive timing but loses bar structure, metrical position, and tempo context. Long pauses generate many consecutive TimeShift tokens.

**REMI** (Huang & Yang, 2020, arXiv:2008.07009) replaces time shifts with position-based metrical structure. Token types: `Bar_None` (new bar), `Position_{p}` (0–31 for 32 subdivisions per bar), `Pitch_{p}` (21–109 piano range or 0–127 full range), `Velocity_{v}` (16 or 32 bins), `Duration_{beats.subbeats.ticks}` (50–200 tokens), `Tempo_{bpm}` (32 bins spanning 40–250 BPM), and optional `Chord_{root}_{quality}`. Time quantization: with `beat_res = {(0,4): 8, (4,12): 4}`, beats 0–4 get 8 subdivisions per beat (32 positions in a 4/4 bar). **Vocabulary: ~200–500 tokens. Average sequence: ~1,701 tokens.** Preserves bar structure, metrical position, tempo, and chord context. Loses sub-beat expressive timing (quantized to grid).

**TSD (Time Shift Duration)** is similar to MIDILike but uses explicit Duration tokens instead of NoteOff. Token types: Pitch, Velocity, Duration, TimeShift, optional Program. Duration tokens are more informative for causal generation since the model knows duration at note onset time. Same long-pause issue as MIDILike.

**Structured** (Hadjeres & Crestel, 2021) enforces a strict recurring pattern: `Pitch → Velocity → Duration → TimeShift → Pitch → ...` Simultaneous notes use `TimeShift_0`. This rigid structure simplifies training — the model always knows what token type to predict next. Average sequence: ~1,791 tokens.

**CPWord (Compound Word)** (Hsiao et al., 2021) groups multiple token types into a single compound step. Each compound token is a tuple: (Family, Position/Bar, Pitch, Velocity, Duration, Tempo, Chord). Individual embeddings are computed per sub-token, then summed/concatenated and projected to a single vector. **Reduces sequence length to ~806 tokens (~50% of REMI)**, which directly impacts transformer attention cost at O(n²). Requires multiple output heads during training.

**Octuple/OctupleMIDI** (Zeng et al., MusicBERT, ACL 2021, arXiv:2106.05630) represents each note as an 8-tuple: (TimeSignature, Tempo, Bar, Position, Instrument, Pitch, Duration, Velocity). Eight separate embedding tables concatenate into a single vector; eight separate softmax layers decode each element. Sequence length drops to **~400–600 tokens** — the most compact representation. Best for music understanding/classification tasks (MusicBERT). Generation requires sampling from 8 distributions per step.

### 1.2 Piano roll representations

A piano roll is a binary or real-valued matrix of shape **(128, T)** where 128 = MIDI pitches and T = time steps. Resolution is determined by frames per second (fs=100 → 10ms resolution) or beat-aligned grids (16th-note = 16 steps per bar in 4/4). Cell values can be binary (1 = note active), velocity-weighted (0–127), or onset-only. A 4-bar phrase at 16th-note resolution: T = 64 columns, matrix = 128 × 64 = 8,192 values — extremely sparse (~95%+ zeros). Generated via `pretty_midi.get_piano_roll(fs=100)` or the dedicated `pypianoroll` library. Best for CNN/GAN-based models (MuseGAN), visualization. Not ideal for transformers.

### 1.3 MusicXML and ABC notation tokenization

**MusicXML** carries far more information than MIDI: key signatures, time signatures, dynamics markings, articulations, slurs, ties, repeats, lyrics, part names, clef changes, grace notes, tuplets, and beaming. It's ~10× larger than equivalent MIDI. Typically parsed via `music21` and converted to MIDI or a simpler representation for ML. Some score-based models work with tokenized MusicXML directly.

**ABC notation** is compact, text-based, and widely used for folk music. Notes are letter names with accidentals (^ = sharp, _ = flat), duration modifiers (2 = double, / = half), and barlines. For AI: character-level tokenization is simple but creates long sequences. BPE tokenization is used by **MuPT** (Qu et al., 2024) on Synchronized Multi-Track ABC notation. **NotaGen** (2025) uses ABC notation with BPE, pretrained on 1.6M sheets. Preserves key signature, meter, and note spelling (C# vs Db) that MIDI lacks. No velocity information.

### 1.4 Token vocabularies: BPE, Unigram, custom designs

**BPE for music** treats base music tokens (~200–500) as bytes and iteratively merges the most frequent adjacent pairs. Applied to REMI tokens: 17 base tokens → 10 BPE tokens (41% compression). Sequence lengths reduce by **30–60%** depending on vocab size. Fradet et al. (2023) found optimal BPE vocab sizes around **10k–20k for ~40M parameter models**, improving generation quality AND inference speed. Most learned tokens represent single notes (Pitch+Velocity+Duration combos). Implementation via MiDiTok using HuggingFace `tokenizers` Rust library.

**Unigram LM** starts with a large vocabulary and prunes tokens that least increase corpus loss. MiDiTok supports Unigram and WordPiece via HuggingFace tokenizers. Early evidence suggests Unigram may outperform BPE for symbolic music by better preserving musical structure.

**Vocab size impact**: Base vocab (~200–500) underutilizes embedding space. BPE 5k–10k is the sweet spot. BPE 20k+ shows diminishing returns requiring larger models and more data.

### 1.5 Key libraries

**MiDiTok** (github.com/Natooz/MidiTok, pip install miditok): Supports REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM, PerTok. BPE/Unigram/WordPiece training. HuggingFace Hub integration. Data augmentation. PyTorch DataLoader support.

```python
from miditok import REMI, TokenizerConfig
config = TokenizerConfig(pitch_range=(21, 109), beat_res={(0, 4): 8, (4, 12): 4},
                         num_velocities=32, use_chords=True, use_tempos=True)
tokenizer = REMI(config)
tokens = tokenizer(Score("file.mid"))  # TokSequence with .ids, .tokens
tokenizer.train(vocab_size=30000, files_paths=midi_paths)  # BPE training
```

**pretty_midi** (github.com/craffel/pretty-midi): MIDI I/O with notes as (start_seconds, end_seconds, pitch, velocity). Key functions: `estimate_tempo()`, `get_beats()`, `get_chroma()`, `get_piano_roll(fs)`. No built-in ML tokenization.

**music21** (github.com/cuthbertLab/music21): Comprehensive toolkit for musicology. Reads MusicXML, MIDI, ABC, Humdrum. Key detection, chord analysis, Roman numeral analysis, pitch class set theory. Built-in corpus of thousands of pieces. The standard tool for parsing and analyzing scores before tokenization.

### 1.6 Key papers

**Music Transformer** (Huang et al., ICLR 2019, arXiv:1809.04281): First successful transformer for music with long-term structure (~60s). Memory-efficient relative attention with "skewing" procedure reducing memory from O(L²D) to O(LD). MIDILike tokenization, ~388 tokens, MAESTRO dataset. NLL 1.835 on Piano-e-Competition.

**MusicBERT** (Zeng et al., ACL 2021, arXiv:2106.05630): OctupleMIDI encoding, bar-level masking strategy, Million MIDI Dataset (1M+ songs). State-of-the-art on melody completion, accompaniment suggestion, genre/style classification. GitHub: github.com/microsoft/muzic.

**PopMAG** (Ren et al., ACM MM 2020, arXiv:2008.07703): MuMIDI representation encoding multiple tracks with inter-track dependencies. Transformer-XL backbone. Melody-conditioned accompaniment generation.

**MMM** (Ens & Pasquier, 2020, arXiv:2008.06048): Track-concatenation representation with track delimiters. Bar-level inpainting via `Bar_Fill` tokens. Attribute control for instrumentation and note density. Supports arbitrary number of tracks. Implemented in MidiTok.

| Tokenization | Tokens/Note | Avg Seq Length | Multi-track | Best For |
|---|---|---|---|---|
| MIDILike | ~3–4 | ~1,772 | Limited | Expressive piano |
| REMI | ~5 | ~1,701 | REMI+ | Pop generation |
| CPWord | 1 compound | ~806 | Yes | Efficiency |
| Octuple | 1 compound (8-tuple) | ~400–600 | Yes | Understanding tasks |
| MMM | ~3–4 per track | Varies | Yes (concatenated) | Inpainting/editing |

---

## 2. Audio and spectral tokenization

### 2.1 Raw waveform representations

**PCM (Pulse Code Modulation)**: 16-bit signed integer [-32768, 32767] at 44.1kHz (CD standard) yields ~10.6 million samples per 4-minute mono song (~41 MB stereo). For neural network input, converted to float32 in [-1.0, 1.0] via `librosa.load()` or `torchaudio.load()`.

**WaveNet μ-law quantization** compresses 16-bit audio to 256 categorical bins: `f(x) = sign(x) × ln(1 + μ|x|) / ln(1 + μ)` where μ = 255 and x ∈ [-1, 1]. Non-linear compression allocates more levels near zero amplitude where human hearing is more sensitive. Each sample becomes a one-hot vector of size 256. Reconstruction quality: ~30 dB SNR. WaveNet generates one sample at a time — **~16,000 forward passes per second at 16kHz**, originally ~1000× slower than real-time. Parallel WaveNet achieved 20× faster than real-time via knowledge distillation. For higher fidelity, discretized mixture of logistics (K=10) models 16-bit audio directly.

### 2.2 STFT (Short-Time Fourier Transform)

The STFT of signal x[n] at frame m is:

```
X[m, k] = Σ_{n=0}^{N-1} x[n + m·H] · w[n] · e^{-j2πkn/N}
```

where w[n] is the window function (Hann: `w[n] = 0.5(1 - cos(2πn/(N-1)))`), N = window size (typically **2048**), H = hop size (typically **512 = N/4**), k = frequency bin (0 to N/2). Output is complex-valued: magnitude `|X[m,k]|` and phase `φ[m,k] = atan2(Im(X), Re(X))`.

**Output shape**: (n_fft/2 + 1, n_time_frames) = e.g., **(1025, 862)** for 5s at 22050Hz. Each cell represents the complex amplitude of frequency bin k at time frame m. Frequency resolution = sr/n_fft (44100/2048 ≈ 21.5 Hz per bin). Time resolution = hop/sr (512/44100 ≈ 11.6 ms per frame).

**Phase reconstruction via Griffin-Lim**: When only magnitude is available (common after spectrogram modification), initialize random phase, then iterate: construct complex spectrogram → inverse STFT → forward STFT → update phase. Typically 32–1000 iterations. Fast Griffin-Lim adds momentum α ≈ 0.99. Produces audible artifacts — **modern neural vocoders (HiFi-GAN, WaveGlow) are preferred**.

```python
stft_complex = librosa.stft(y, n_fft=2048, hop_length=512)  # (1025, n_frames), complex64
magnitude = np.abs(stft_complex)
log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
y_reconstructed = librosa.griffinlim(magnitude, n_iter=32, hop_length=512)
```

**Preserved**: Complete frequency content within Nyquist, temporal evolution, magnitude. **Lost**: Phase (when taking magnitude), absolute timing precision below hop_length.

### 2.3 Mel-spectrogram

The mel scale maps frequency to perceived pitch: **m = 2595 × log₁₀(1 + f/700)**. Approximately linear below 1000 Hz, logarithmic above. A mel filterbank constructs n_mels triangular bandpass filters at mel-spaced center frequencies. Typical n_mels: **64** (lightweight), **80** (speech/TTS standard), **128** (music, higher resolution).

**Pipeline**: waveform → STFT(n_fft=2048, hop=512) → |magnitude|² → mel filterbank multiplication → log₁₀(·) → mel-spectrogram. Output shape: **(n_mels, n_frames)** — e.g., (128, 216) for 5s at 22050Hz.

```python
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)  # shape: (128, n_frames)
```

**Gained**: Perceptual alignment with human hearing, dimensionality reduction (128 vs 1025 bins). **Lost**: Fine frequency detail at high frequencies (irreversible many-to-one), phase. Used by Jukebox, audio diffusion models (AudioLDM, Riffusion), Tacotron2. **MelTok** (2025) encodes 44.1kHz audio into mel-spectrogram tokens with a single codebook. Computational cost is low: O(N log N) for STFT + O(n_mels × n_fft/2) matrix multiply per frame.

### 2.4 Constant-Q Transform (CQT)

The CQT uses **logarithmically spaced frequency bins** with constant Q factor (ratio of center frequency to bandwidth). Center frequencies follow a geometric series: **f_k = f_min × 2^{k/B}** where B = bins per octave. Q factor: **Q = 1 / (2^{1/B} - 1)** — for B=12, Q ≈ 16.82. Window length varies per bin: N_k = ceil(Q × f_s / f_k). Low-frequency bins get long windows (high frequency resolution), high-frequency bins get short windows (high time resolution).

**Why CQT is better for music**: With B=12, each bin corresponds exactly to one semitone. Octave transposition = shift along frequency axis. Harmonics form a constant pattern regardless of fundamental frequency.

```python
C = librosa.cqt(y, sr=22050, hop_length=512, fmin=librosa.note_to_hz('C1'),
                n_bins=84, bins_per_octave=12)  # 7 octaves, shape (84, n_frames)
```

| bins_per_octave | Resolution | Use Case |
|---|---|---|
| 12 | Semitone | Chord recognition, standard music analysis |
| 24 | Quarter-tone | Fine pitch analysis, non-Western music |
| 36 | 1/3 semitone | High-resolution pitch tracking |

**Preserved**: Harmonic structure, musical pitch, timbral patterns invariant to transposition. **Lost**: DC component, linear frequency detail. Cost higher than STFT — O(L log L) per octave.

### 2.5 Chroma features and pitch class profiles

Chroma features project spectral energy onto the **12 pitch classes** (C, C#, D, ..., B). Computed by folding CQT or STFT bins across octaves into a 12-dimensional vector per frame. Shape: **(12, n_frames)**.

```python
chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, n_octaves=7)
```

**Direct link to Krumhansl-Schmuckler key detection**: The algorithm correlates the average chroma vector against predefined key profiles for all 24 major/minor keys via Pearson correlation:

```
R(d, p) = Σᵢ(dᵢ - d̄)(pᵢ - p̄) / √[Σᵢ(dᵢ - d̄)² × Σᵢ(pᵢ - p̄)²]
```

**Major profile**: [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]. **Minor profile**: [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]. For each of 24 keys, rotate the profile to align with the candidate tonic, compute R. Highest R wins. Alternative profiles: Temperley-Kostka-Payne, Aarden-Essen, Bellman-Budge.

**Connection to Camelot Wheel**: Chroma → K-S → key → Camelot code. Tools like Mixed In Key and Essentia use this pipeline. **Preserved**: Tonal/harmonic content, key, chord progressions. **Lost**: Octave info, timbre, dynamics, rhythm.

### 2.6 MFCCs (Mel-Frequency Cepstral Coefficients)

**Pipeline**: signal → pre-emphasis (y[n] = x[n] - 0.97·x[n-1]) → framing → windowing → FFT → |·|² → mel filterbank → log(·) → DCT → MFCCs. Output: **(n_mfcc, n_frames)**.

What each coefficient encodes: **c₀** = average log energy; **c₁** = broad spectral tilt (spectral centroid); **c₂–c₅** = spectral envelope / timbre; **c₆–c₁₃** = finer spectral detail; **>c₁₃** = noise-sensitive, typically discarded. Typical n_mfcc: 13 (speech), 20 (enhanced), 40 (high-resolution). Good for classification (genre, speaker, instrument). Bad for generation — DCT discards fine structure, phase absent, inverting MFCCs produces severe quality loss. Modern trend favors raw mel spectrograms or learned representations.

### 2.7 Onset and beat tokenization

**Spectral flux** (most common onset detector): `SF[m] = Σ_k max(0, |X[m,k]| - |X[m-1,k]|)`. **SuperFlux** extends with maximum filtering for vibrato suppression. Neural methods (madmom's CNNOnsetProcessor) achieve F-measures >85%.

**Beat tracking**: librosa uses onset strength → autocorrelation → dynamic programming. madmom's RNNBeatProcessor (bidirectional RNN) + DBNBeatTrackingProcessor achieves state-of-the-art accuracy.

```python
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
```

Key libraries: **librosa** (STFT, mel, CQT, chroma, MFCC, beats), **torchaudio** (GPU-accelerated, differentiable), **madmom** (SOTA onset/beat detection, pre-trained models), **essentia** (production-grade, used by Spotify), **nnAudio** (Conv1D-based differentiable transforms).

---

## 3. Learned and neural tokenization

### 3.1 VQ-VAE (Vector Quantized VAE)

**Paper**: van den Oord et al., NeurIPS 2017 (arXiv:1711.00937). Architecture: encoder maps input x to latent z_e(x), then **quantization**: z_q = e_k where k = argmin_j ||z_e(x) - e_j||₂ (nearest neighbor in codebook of K embedding vectors). Decoder reconstructs from z_q.

**Loss function**: `L = L_reconstruction + ||sg[z_e] - e_k||₂² + β·||z_e - sg[e_k]||₂²` where sg = stop-gradient, β typically 0.25. Gradients through non-differentiable quantization use the **Straight-Through Estimator** — gradients from decoder copied directly to encoder. Typical codebook sizes: K = 512, 1024, 2048, 8192 with embedding dimension D = 64 or 128.

**Jukebox** (OpenAI, 2020, arXiv:2005.00341) uses three VQ-VAE modules at different temporal scales: bottom (8× compression, 5512.5 tokens/sec), middle (32×, 1378 tokens/sec), top (128×, **344.5 tokens/sec**). Codebook K=2048, D=64 at each level. Prior models: autoregressive transformers up to **5B total parameters**. Multi-resolution STFT spectral loss critical for preserving high frequencies. Generation: **~9 hours per minute** of music on V100. GitHub: github.com/openai/jukebox (MIT).

### 3.2 EnCodec (Meta, 2022)

**Paper**: Défossez et al., arXiv:2210.13438. Architecture: SEANet-based 1D convolutional encoder → **Residual Vector Quantization (RVQ)** → symmetric decoder. Novel Multi-Scale STFT Discriminator.

**RVQ explained**: Cascade of VQ layers where each quantizes the residual from the previous:

```
r_0 = z_e (encoder output)
For i = 1 to N_cb:
    q_i = VQ_i(r_{i-1})     # quantize residual
    r_i = r_{i-1} - q_i     # compute new residual
z_q = Σ(q_1, ..., q_{N_cb})  # final quantized
```

First codebook captures coarse structure; later codebooks refine with higher-frequency detail. **Codebook size: 1024 entries = 10 bits per codebook.**

| Bandwidth | Codebooks | Tokens/sec (24kHz, 75Hz frame rate) |
|---|---|---|
| 1.5 kbps | 2 | 150 |
| 3.0 kbps | 4 | 300 |
| 6.0 kbps | 8 | 600 |
| 12.0 kbps | 16 | 1,200 |
| 24.0 kbps | 32 | 2,400 |

**Bitrate formula**: `bitrate = N_codebooks × log₂(codebook_size) × frame_rate`. Example: 8 × 10 × 75 = 6,000 bps. At 3 kbps, outperforms Opus at 12 kbps. Optional causal transformer for 25–40% lossless entropy coding savings.

```python
from transformers import EncodecModel, AutoProcessor
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
inputs = processor(raw_audio=audio, sampling_rate=24000, return_tensors="pt")
enc_out = model.encode(inputs["input_values"], inputs["padding_mask"])
# enc_out.audio_codes shape: [B, n_codebooks, T]
decoded = model.decode(enc_out.audio_codes, enc_out.audio_scales)
```

GitHub: github.com/facebookresearch/encodec (MIT). Also on HuggingFace: `facebook/encodec_24khz`, `facebook/encodec_48khz`.

### 3.3 DAC (Descript Audio Codec, 2023)

**Paper**: Kumar et al., arXiv:2306.06546. Key improvements over EnCodec: **snake activation** (`x + (α+ε)⁻¹ · sin²(αx)`) providing periodic inductive bias for waveform generation; improved codebook learning via L2 normalization + factorized codes converting Euclidean to cosine similarity (dramatically better codebook utilization); multi-scale discriminator (MSD + MPD from HiFi-GAN); quantization dropout (rate 0.5) for flexible bitrate.

**44kHz model**: 9 codebooks, 1024 entries each, codebook dim 8, encoder strides [2,4,8,8] → total stride **512**. Token rate: 44100/512 ≈ **86 Hz per codebook**. Bitrate: ~8 kbps (~90× compression). **DAC at 8 kbps outperforms EnCodec at 24 kbps** on ViSQOL, SI-SDR, and perceptual metrics.

```python
import dac
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path).to('cuda')
x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)
y = model.decode(z)
```

GitHub: github.com/descriptinc/descript-audio-codec (MIT). HuggingFace: `descript/dac_44khz`.

### 3.4 SoundStream (Google, 2021)

**Paper**: Zeghidour et al., IEEE/ACM TASLP 2022, arXiv:2107.03312. **First end-to-end neural audio codec** — jointly trains encoder, RVQ, and decoder. SEANet-based encoder with strides [2,4,5,8] → total stride 320 → **75 Hz at 24kHz**. Introduced **quantizer dropout** for variable bitrate (single model operates at 3–18 kbps). At 3 kbps, outperforms Opus at 12 kbps. ~27M parameters. Real-time on smartphone CPU. Integrated into Google's Lyra codec. Not open-source from Google; community implementations at github.com/wesbz/SoundStream and in `lucidrains/audiolm-pytorch`.

### 3.5 MusicGen (Meta, 2023)

**Paper**: Copet et al., NeurIPS 2023, arXiv:2306.05284. Uses **32kHz EnCodec** producing **4 codebooks with 2048 entries at 50 Hz**. A transformer LM generates codebook tokens autoregressively, decoded back via EnCodec.

**The delay pattern** avoids multiplying sequence length by N_q. Each codebook is delayed by 1 step:

```
Time step:  t1  t2  t3  t4  t5
Codebook 1: c1  c2  c3  c4  c5
Codebook 2:  0  c1  c2  c3  c4
Codebook 3:  0   0  c1  c2  c3
Codebook 4:  0   0   0  c1  c2
```

At each autoregressive step, all codebooks predicted in parallel but offset → only **50 steps/second** (not 200). Text conditioning via T5/FLAN-T5 cross-attention. Melody conditioning via extracted chromagram (12 bins) with information bottleneck. Model sizes: **300M, 1.5B, 3.3B** parameters. Trained on 20K hours of licensed music.

```python
from audiocraft.models import MusicGen
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=8)
wav = model.generate_with_chroma(descriptions=["electronic remix"], 
    melody_wavs=source_tensor, melody_sample_rate=sr)
```

GitHub: github.com/facebookresearch/audiocraft (MIT code, CC-BY-NC weights).

### 3.6 AudioLM (Google, 2022)

**Paper**: Borsos et al., IEEE/ACM TASLP 2023, arXiv:2209.03143. Dual tokenization: **semantic tokens** from w2v-BERT (0.6B Conformer, k-means K=1024, **25 Hz**) capture long-term structure (language, melody, harmony) but reconstruct poorly; **acoustic tokens** from SoundStream (RVQ Q=12, 1024 vocab, **50 Hz**, split into coarse Q'=4 and fine Q-Q'=8) capture audio quality but lack coherence alone. Combined: both structure AND fidelity.

**Three-stage generation** (each a decoder-only transformer):

| Stage | Output | Purpose |
|---|---|---|
| Semantic | Future semantic tokens | Long-term structure |
| Coarse acoustic | Coarse tokens (4 codebooks) | Timbre/characteristics |
| Fine acoustic | Fine tokens (8 codebooks) | High-frequency detail |

Human evaluators: 51.2% accuracy distinguishing AudioLM speech from real (near chance). Community reimplementation: github.com/lucidrains/audiolm-pytorch.

### 3.7 MusicLM (Google, 2023)

**Paper**: Agostinelli et al., arXiv:2301.11325. Extends AudioLM for music with MuLan text conditioning.

**Three-level token hierarchy**: Semantic (w2v-BERT, K=1024, **25 tokens/sec**) → Coarse acoustic (SoundStream first 4 RVQ levels, **200 tokens/sec**) → Fine acoustic (remaining 8 levels, **400 tokens/sec**). Total: **625 tokens per second**. MuLan embeddings (128-d joint audio-text space, trained on 44M recordings) quantized via RVQ as conditioning tokens. At training, MuLan audio embeddings condition generation; at inference, MuLan text embeddings substitute (shared embedding space enables this swap). Three ~430M parameter transformers (~1.3B total). Community: github.com/lucidrains/musiclm-pytorch.

### 3.8 Cutting-edge codecs (2024–2026)

**WavTokenizer** (ICLR 2025): Extreme compression — **single quantizer, 40–75 tokens/sec** at 24kHz. ~430M parameters, VOCOS-based decoder. State-of-the-art UTMOS scores.

**DisCodec** (NeurIPS 2024 Workshop): High-fidelity 44.1kHz codec with ConvNeXt + attention layers, affine re-parametrization. GitHub: github.com/ETH-DISCO/discodec.

**Key trends**: Single-codebook models replacing multi-codebook RVQ; semantic-acoustic boundary blurring; continuous latent diffusion (Stable Audio) competing with discrete autoregressive approaches; universal codecs spanning speech, music, and effects.

| System | Year | Codebook Size | Codebooks | Token Rate | Bitrate | Open Source |
|---|---|---|---|---|---|---|
| Jukebox | 2020 | 2048 | 1 per level (3) | 345–5512 Hz | — | github.com/openai/jukebox |
| SoundStream | 2021 | 1024 | 3–80 | 50–75 Hz | 3–18 kbps | Community only |
| EnCodec | 2022 | 1024 | 2–32 | 75 Hz (24k) | 1.5–24 kbps | github.com/facebookresearch/encodec |
| DAC | 2023 | 1024 | 9 | 86 Hz (44.1k) | ~8 kbps | github.com/descriptinc/descript-audio-codec |
| MusicGen | 2023 | 2048 | 4 | 50 Hz | ~2.2 kbps | github.com/facebookresearch/audiocraft |
| WavTokenizer | 2025 | large | 1 | 40–75 Hz | ultra-low | Available |

---

## 4. Harmonic and music-theory-aware tokenization

### 4.1 Chord tokenization

**Roman numeral tokens** encode chords relative to key: I, ii, iii, IV, V, vi, vii°. The MusicLang tokenizer implements this with factorized tokens: `CHORD_DEGREE`, `TONALITY_DEGREE`, `TONALITY_MODE`, `CHORD_EXTENSION`. Math: for chord root R in key K, Roman numeral = `(R - K) mod 12` mapped to scale degree.

**Chord symbol vocabularies** range widely. Flat vocabulary (each chord a unique token): the iReal Pro dataset yields **5,202 unique chord tokens**. Factorized vocabulary (root + quality as separate tokens): reduces to ~59 chord-specific tokens. Typical practical range: **12 roots × 30 quality types = 360 possible chords**. Factorized representation: `chord_id = root × num_qualities + quality_id`.

**HarmonyTok** (2025) compares four tokenization strategies: full chord symbols, root + quality separated, pitch-class sets, and pitch-class sets with designated root. Finding: spelling-based methods achieve higher accuracy and lower perplexity; chunky methods produce more stylistically faithful harmonizations.

Libraries: **madmom** (deep learning chord recognition), **Chordino** (HMM-based from audio), **MiDiTok** (rule-based from MIDI with `use_chords=True`), **music21** (Roman numeral analysis).

### 4.2 Key-conditioned representations

**Key as conditioning token**: MusicLang uses explicit tokens: `TONALITY_DEGREE__n` (0–11) + `TONALITY_MODE__M/m`. MiDiTok supports `KeySignature` tokens. FIGARO and REMI+ include key signature as a global conditioning token.

**Transposition-invariant representations**: Instead of absolute MIDI notes, encode pitches relative to current key root. Math: `relative_pc = (P - K) mod 12`. MusicLang encodes notes as scale degrees (s0, s1, s2...) making sequences key-invariant — dramatically improves BPE efficiency. Music Transformer uses random transposition ±6 semitones for training augmentation.

### 4.3 Camelot Wheel as tokenization constraint

The Camelot Wheel maps 24 keys to positions: numbers 1–12 (following circle of fifths) × letters A (minor) / B (major). Moving +1 = perfect fifth (7 semitones). Relative major/minor pairs share the same number.

| Camelot | Key | Camelot | Key |
|---|---|---|---|
| 1A | Ab minor | 1B | B major |
| 5A | C minor | 5B | Eb major |
| 8A | A minor | 8B | C major |
| 12A | C# minor | 12B | E major |

**Compatible mixing rules**: Same key = 1.0 compatibility. ±1 same letter = 6/7 shared notes (~0.86). Same number, switch letter = all 7 notes shared (1.0). ±2 same letter = 5/7 shared notes (~0.71).

**As loss function**: `L_camelot = -log P(next_key | current_key) × (1 - compatibility(current, next))`. **As embedding**: encode 24 positions on a circle: `angle = (camelot_num - 1) × (2π / 12)`, x = cos(angle), y = sin(angle), z = 0 if major else 1. **Distance**: `fifths_distance = min(|c1 - c2|, 12 - |c1 - c2|)`.

### 4.4 Tonal Pitch Space (Lerdahl)

Fred Lerdahl's *Tonal Pitch Space* (Oxford, 2001) provides a hierarchical, quantitative model. Four levels for C major: chromatic (all 12 notes) → diatonic (C D E F G A B) → triadic (C E G) → root (C). Root = most stable; chromatic-only = least stable.

**Chord distance within a key**: `δ(x → y) = j + k` where j = steps on chord/fifths circle, k = distinctive pitch classes. **Across keys**: `d(chord) = i + j + k` where i = steps on region circle. Example: I to V in C major: j=1, k=1 → δ = 2. I to vi: j=3, k=0 → δ = 3.

For ML: TPS distances serve as pre-computed token distance matrices for loss functions, initialization for chord embedding spaces, or regularization penalizing transitions proportional to TPS distance. Burg & Serafin embedded TPS into low-dimensional vector spaces preserving the toroidal structure.

### 4.5 Interval vectors and pitch class set theory

**Pitch class sets** are subsets of {0, 1, ..., 11} represented as 12-bit binary vectors. C major triad {0, 4, 7} → [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0].

**Interval class**: `ic(a, b) = min(|a - b| mod 12, 12 - |a - b| mod 12)` — values in {0, ..., 6}. Maps inversionally equivalent intervals: m2/M7→1, M2/m7→2, m3/M6→3, M3/m6→4, P4/P5→5, tritone→6.

**Interval vector** (6-element): count of each interval class among all pairs. Major triad: ⟨001110⟩. Dom7: ⟨012111⟩. Major scale: ⟨254361⟩ (deep scale). Major and minor triads share the same IV — they are inversionally related.

**Allen Forte's catalog**: ~220 distinct set classes across cardinalities 3–9. Forte numbers (e.g., "3-11" for any major/minor triad) serve as discrete tokens. Z-related sets share the same IV but aren't T/I equivalent. Libraries: `music21` (comprehensive pc set analysis).

### 4.6 Beat and tempo as time-step tokens

**Tempo tokens in MiDiTok**: configurable bins. Linear binning: `bin_i = round((BPM - BPM_min) / (BPM_max - BPM_min) × (num_bins - 1))`. Log binning better captures perceptual differences (60–70 BPM difference matters more than 190–200 BPM). Default: 32 bins, range 40–250 BPM.

**Time signature tokens**: `TimeSignature_4/4`, `3/4`, `6/8`, etc. Placed at bar beginnings in REMI, only at changes in TSD. Typical 5–15 supported signatures.

**Beat position tokens**: REMI uses Bar + Position tokens. With beat_res=8 in 4/4: 32 positions per bar. Position_0 = downbeat, Position_8 = beat 2, Position_16 = beat 3, Position_24 = beat 4. **Metric hierarchy**: bar → beat → subdivision (tatum). Swing quantization shifts alternate subdivisions from 50/50 to ~67/33 (triplet feel).

---

## 5. Multimodal and structured tokenization

### 5.1 Joint audio + text tokens (CLAP and MuLan)

**CLAP** (Contrastive Language-Audio Pretraining) maps audio and text into a shared embedding space via symmetric InfoNCE contrastive loss. **LAION CLAP** (Wu et al., ICASSP 2023): HTS-AT audio encoder (Swin-Transformer), RoBERTa text encoder, **512-d shared space**, trained on LAION-Audio-630K (633K pairs). HuggingFace: `laion/clap-htsat-fused`. **Microsoft CLAP**: CNN14 audio + BERT text encoders, **1024-d shared space**, 128K pairs. Extensions include T-CLAP (temporal), tinyCLAP (>90% compression), CoLLAP (long-form 5-min audio), CLaMP 3 (universal cross-modal, ACL 2025).

**MuLan** (Google, ISMIR 2022, arXiv:2208.12415): Music-specific two-tower model. ResNet-50 audio tower + BERT text tower → **128-d shared space**. Trained on **44M music recordings** (370K hours) with weakly-associated text — massively larger than CLAP. Not publicly released. Used in MusicLM: MuLan embeddings quantized via RVQ as conditioning tokens.

**Practical flow**: `text description → CLAP/MuLan text encoder → embedding vector → quantize (optional) → cross-attention conditioning → generative model`.

### 5.2 Beat-synchronized tokenization

Audio features aligned to beat grid: run beat tracker → get beat timestamps → extract features per beat interval → one feature vector per beat. Advantages for remix/DJ: tokens snap to musical time, tempo-independent comparison, phase alignment, beat-grid-quantized editing. Beat-synced tokenization is **invariant to time-stretching** — the token sequence stays the same regardless of playback tempo.

```python
beat_frames = librosa.beat.beat_track(y=y, sr=sr)[1]
chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
```

### 5.3 Hierarchical tokenization

**Bar-level**: Captures overall structure and form. **MuseTok** (ICASSP 2026) uses RQ-VAE to learn discrete bar-level representations from REMI+ events. **Bar Transformer** (Qin et al., 2023) uses note-level encoder within bars + bar-level encoder across bars. **Beat-level**: Rhythmic patterns, chord changes. REMI's Bar + Position tokens. **Note-level**: Individual pitch/velocity/duration events.

Benefits: long-range coherence at bar level (O(32²) attention for 32 bars vs O(2000²) for notes), fine detail at note level. **Nested Music Transformer** (2024) bridges compound and flat: main decoder for compound tokens + sub-decoder for sub-tokens within each compound.

### 5.4 Compound tokens vs flat tokens

**Flat**: Each attribute as separate token → long sequences. REMI: Bar → Position → Pitch → Velocity → Duration. **Compound**: Multiple attributes packed into single token. CPWord reduces to ~50% of REMI length. Octuple reduces to ~15–25%. Impact on attention cost: REMI 2-min piece ~2000–4000 tokens → ~4M–16M attention operations; Octuple ~300–600 tokens → ~90K–360K operations. **10–50× speedup** in attention computation.

### 5.5 Cross-modal representations

**CAMIL** (EUSIPCO 2025): Contrastive framework aligning MIDI and audio spectrograms in shared embedding space. **CLaMP 3** (ACL 2025): Universal cross-modal learning across sheet music, symbolic scores, audio, and text covering 100+ languages. **Unified Cross-modal Translation** (2025): Single model translating between score images, MIDI, and audio.

### 5.6 Emerging approaches

**Stable Audio Open** (2024, arXiv:2407.14358): Continuous latent diffusion (not discrete tokens) via fully-convolutional VAE, latent rate 21.5 Hz, dim 64. DiT generative model on 1024 latent tokens. T5-base text conditioning. ~1.21B parameters. HuggingFace: `stabilityai/stable-audio-open-1.0`. Represents the continuous-latent alternative to discrete tokenization.

---

## 6. Practical integration for AI RemixMate

### 6.1 Demucs stems + codec tokens pipeline

Demucs htdemucs_ft outputs 4 stems (vocals, drums, bass, other) at input sample rate. Each stem is independently encoded with EnCodec or DAC for per-stem token manipulation.

```python
import torch, torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from transformers import EncodecModel, AutoProcessor

# Demucs separation
demucs = get_model('htdemucs_ft').cuda()
waveform, sr = torchaudio.load("input.wav")
stems = apply_model(demucs, waveform.unsqueeze(0).cuda(), device='cuda')
# stems: [1, 4, channels, samples] → drums, bass, other, vocals
demucs.cpu(); del demucs; torch.cuda.empty_cache()  # free VRAM

# Per-stem EnCodec encoding
encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").cuda()
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
resampler = torchaudio.transforms.Resample(sr, 24000).cuda()
stem_tokens = {}
for i, name in enumerate(['drums', 'bass', 'other', 'vocals']):
    stem_24k = resampler(stems[0, i])
    inputs = processor(raw_audio=stem_24k[0].cpu().numpy(),
                       sampling_rate=24000, return_tensors="pt")
    with torch.no_grad():
        stem_tokens[name] = encodec.encode(inputs["input_values"].cuda(),
                                            inputs["padding_mask"].cuda())
```

**GPU memory**: Demucs ~2 GB, EnCodec ~500 MB. Strategy: sequential model loading, peak ~3–4 GB. **CodecSep** (arXiv:2509.11717) validates operating in codec latent space for separation. AudioCraft's `CompressionModel` API unifies EnCodec and DAC.

### 6.2 Chroma/CQT → Camelot Wheel integration

```python
import librosa, numpy as np

MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
PITCH_CLASSES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CAMELOT = {'C major':'8B','G major':'9B','D major':'10B','A major':'11B',
           'E major':'12B','B major':'1B','F# major':'2B','C# major':'3B',
           'G# major':'4B','D# major':'5B','A# major':'6B','F major':'7B',
           'A minor':'8A','E minor':'9A','B minor':'10A','F# minor':'11A',
           'C# minor':'12A','G# minor':'1A','D# minor':'2A','A# minor':'3A',
           'F minor':'4A','C minor':'5A','G minor':'6A','D minor':'7A'}

def detect_key_and_camelot(audio, sr):
    y_harmonic, _ = librosa.effects.hpss(audio)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, n_chroma=12,
                                         n_octaves=7, bins_per_octave=36)
    chroma_avg = np.mean(chroma, axis=1)
    chroma_avg /= (np.linalg.norm(chroma_avg) + 1e-8)
    best_key, best_corr = None, -1
    for shift in range(12):
        for profile, mode in [(MAJOR_PROFILE,'major'), (MINOR_PROFILE,'minor')]:
            rotated = np.roll(profile, shift)
            rotated /= (np.linalg.norm(rotated) + 1e-8)
            corr = np.corrcoef(chroma_avg, rotated)[0, 1]
            if corr > best_corr:
                best_corr, best_key = corr, f"{PITCH_CLASSES[shift]} {mode}"
    return best_key, CAMELOT[best_key], best_corr

def pitch_shift_for_camelot(source_camelot, target_camelot):
    ns, nt = int(source_camelot[:-1]), int(target_camelot[:-1])
    semi = ((nt - ns) * 7) % 12
    return semi - 12 if semi > 6 else semi
```

Apply per-stem for finer analysis — vocals and bass carry the strongest harmonic signal. Correlation below 0.6 indicates uncertain key.

### 6.3 Generative stem manipulation with current tools

**Proven and working approaches**:

**VampNet** (ISMIR 2023, github.com/hugofloresgarcia/vampnet): Bidirectional transformer over DAC tokens. Masked token modeling with 36 sampling passes. Supports inpainting (generate transitions), outpainting (extend sections), vamping (loop with variation). Directly applicable to remixing: periodic prompts keep every Pth timestep (preserves rhythm), beat-driven prompts unmask beat positions (maintains groove), compression prompts keep first C codebooks (preserves coarse structure), inpaint prompts keep prefix/suffix (generates transitions). **4–6 GB VRAM**.

**MusicGen melody conditioning**: Extracts chromagram from reference → generates new audio matching harmonic structure but with different style per text prompt. This IS practical style transfer.

```python
model = MusicGen.get_pretrained('facebook/musicgen-melody')
wav = model.generate_with_chroma(descriptions=["electronic remix with heavy bass"],
    melody_wavs=source_tensor, melody_sample_rate=sr)
```

**MusiConGen** (github.com/Cyan0731/MusiConGen): MusicGen fine-tuned with temporal chord + BPM conditioning. Accepts explicit chord progressions and BPM.

**Not feasible**: Token arithmetic (adding/subtracting discrete RVQ indices) — codebook indices lack semantically meaningful embedding space. Cross-domain style transfer purely via token manipulation — no system reliably converts genre by token-level operations alone.

### 6.4 Key, tempo, and loudness as conditioning

**Key conditioning**: Text-based (include key in prompt, probabilistic), chromagram-based (MusicGen melody, more reliable), or explicit chord conditioning (MusiConGen). Post-generation verification: re-detect key, reject if mismatch exceeds threshold. For strict key: pitch-shift decoded stems via `pyrubberband`.

**Tempo conditioning**: Include BPM in prompt string, or beat-grid alignment (time-stretch stems to target BPM before generation). MusiConGen provides dedicated BPM input.

**LUFS conditioning**: Measure per-stem loudness with `pyloudnorm` (ITU-R BS.1770-4), normalize to target (-14 LUFS for streaming). Short-term LUFS envelope enables energy-shape analysis and conditioning.

```python
import pyloudnorm as pyln
meter = pyln.Meter(sr)
loudness = meter.integrated_loudness(audio)
normalized = pyln.normalize.loudness(audio, loudness, target_lufs=-14.0)
```

### 6.5 Which tokenization enables which capability

| Capability | Best Tokenization | Tool | Status |
|---|---|---|---|
| Style transfer (change genre, keep melody) | Chromagram + EnCodec RVQ | MusicGen melody | Proven |
| Harmonic analysis and key detection | CQT chroma vectors | librosa + K-S | Proven |
| Camelot harmonic mixing | Pitch class distribution | librosa → Camelot map | Proven |
| Creative inpainting/transitions | DAC RVQ tokens | VampNet | Proven |
| High-fidelity reconstruction | 900 tokens/sec, 44.1kHz | DAC 44kHz (9 codebooks) | Proven |
| Compact representation | 40–75 tokens/sec | WavTokenizer (1 codebook) | Proven |
| Energy/loudness shaping | Waveform domain | pyloudnorm | Proven |
| Token-level gap filling | Single-codebook tokens | Discrete diffusion | Emerging |
| Token arithmetic/blending | N/A | N/A | Not feasible |

**No single tokenization handles all tasks.** AI RemixMate needs a multi-representation pipeline: analytical layer (CQT chroma, LUFS, beat tracking), manipulation layer (EnCodec/DAC tokens), and quality layer (waveform domain for final mixing and mastering).

### 6.6 Recommended architecture and GPU budget

```
INPUT AUDIO (44.1kHz)
  ├── [Analysis] CPU ── librosa tempo, chroma_cqt → K-S → Camelot, pyloudnorm LUFS
  ├── [Separation] GPU ~2GB ── Demucs htdemucs_ft → 4 stems, then offload
  ├── [Tokenization] GPU ~1GB ── Per-stem EnCodec 32kHz (MusicGen-compatible)
  ├── [Remix Logic] GPU on-demand ──
  │     Camelot compatibility, BPM alignment, pitch-shift calculation
  │     VampNet inpainting (~4-6GB) / MusicGen style transfer (~8GB)
  └── [Decode + Master] GPU + CPU ── Per-stem decode, sum, LUFS normalize
```

| Component | VRAM | When |
|---|---|---|
| Demucs htdemucs_ft | ~2 GB | Stage 2, offload after |
| EnCodec 32kHz | ~500 MB | Stages 3+5, keep loaded |
| MusicGen melody (1.5B) | ~8 GB | On-demand |
| VampNet | ~4–6 GB | On-demand |
| Analysis (librosa, pyloudnorm) | CPU only | Always |

**Minimum GPU**: 4 GB (basic separation + tokenization). **Recommended**: 12–16 GB (full pipeline with MusicGen). **Optimal**: 24 GB (concurrent models).

**FastAPI integration**: Use `BackgroundTasks` for GPU operations, `run_in_executor` for CPU-bound analysis, Server-Sent Events for progress. Model locking via `asyncio.Lock` — single GPU op at a time. Lazy model loading with VRAM offload between stages. For production: Celery/Redis queue.

### Key open-source projects for the stack

- **AudioCraft** (github.com/facebookresearch/audiocraft): EnCodec + MusicGen framework
- **Demucs v4** (github.com/adefossez/demucs): 9.2 dB SDR stem separation
- **VampNet** (github.com/hugofloresgarcia/vampnet): Masked acoustic token modeling
- **MusiConGen** (github.com/Cyan0731/MusiConGen): Chord + BPM conditioned MusicGen
- **DAC** (github.com/descriptinc/descript-audio-codec): Highest-fidelity 44kHz codec
- **MiDiTok** (github.com/Natooz/MidiTok): All MIDI tokenization schemes + BPE
- **pyloudnorm** (github.com/csteinmetz1/pyloudnorm): ITU-R BS.1770-4 LUFS
- **madmom** (github.com/CPJKU/madmom): State-of-the-art beat/onset detection

---

## Conclusion

Music tokenization for AI spans three complementary paradigms that AI RemixMate should layer together. **Classical signal processing** (CQT chroma, mel spectrograms, beat tracking) provides the analytical foundation — cheap, deterministic, and directly feeding Krumhansl-Schmuckler key detection and Camelot Wheel logic already in the stack. **Symbolic tokenization** (REMI, CPWord, Octuple via MiDiTok) offers the most musically structured representation and is essential if the project expands into MIDI-domain generation, with BPE at 10k–20k vocab providing the best compression-quality tradeoff. **Neural codecs** (EnCodec at 6 kbps, DAC at 8 kbps) are the critical bridge — they convert Demucs stems into manipulable discrete tokens that generative models like VampNet and MusicGen can process, transform, and regenerate.

The most impactful near-term integration is the **Demucs → per-stem EnCodec encoding → VampNet inpainting** pipeline for creative transitions and fills, paired with **MusicGen melody conditioning** for style transfer. Both are production-quality open source, fit within 12–16 GB VRAM, and compose naturally with the existing Camelot/LUFS mastering chain. The key constraint to internalize is that discrete RVQ codebook indices are not embeddings — token arithmetic doesn't work, and all generative manipulation must go through trained models rather than direct token operations.