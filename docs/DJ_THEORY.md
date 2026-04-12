# DJ Theory Reference
### Research-grade knowledge base for AI RemixMate — derived from professional DJ practice, MIR literature, and the ML frontier

This document is the canonical domain-knowledge reference for every decision the AI mixing engine makes — from Camelot move selection to transition bar counts to EQ strategy. Every constant, weight, and heuristic in the codebase should trace back to a principle documented here.

---

## 1. How DJs Think: The Three-Layer Cognitive Model

Professional DJs process information on three simultaneous timescales during a live set.

| Timescale | Horizon | Focus |
|---|---|---|
| **Micro** | Next 30 s | Beat alignment, EQ balance, fader position, headphone cue |
| **Meso** | Next 2–5 tracks | Energy curve position, crowd response, "mini-set" blocks |
| **Macro** | Entire set | Narrative arc, peak timing, contrast management |

**Implication for AI mixing:** An AI DJ system needs all three layers. Beat tracking and EQ automation cover the micro layer. Compatibility scoring covers the meso layer. Set sequence planning (the hardest unsolved problem) covers the macro layer.

### Set Architecture (2-Hour Template)

| Phase | Time | Energy | Notes |
|---|---|---|---|
| Opening | 0–20 min | 3–5/10 | Curiosity, establish groove, 118–123 BPM |
| First build | 20–40 min | 5–7/10 | Waves of energy |
| First peak | 40–50 min | 7–8/10 | Reward early dancers |
| Valley | 50–65 min | 5–6/10 | Breathing room — "white space" |
| Main build | 65–85 min | 6–8/10 | Tension builds |
| Climax | 85–100 min | 9–10/10 | Set peak — placed ~two-thirds through |
| Resolution | 100–120 min | 6→4/10 | Closure with intention |

**White space rule:** Sustained high energy becomes monotonous without valleys. Valleys make peaks hit harder. Warm-up ratio: mix current energy with higher energy at 5:1.

---

## 2. Harmonic Mixing and the Camelot Wheel

### The System

The Camelot Wheel (Mark Davis / Mixed In Key) remaps the Circle of Fifths to an alphanumeric clock. Numbers 1–12 on a clock face; **A = minor** (inner ring), **B = major** (outer ring). Same-numbered A/B pairs are relative major/minor — they share identical notes.

Moving **+1 clockwise = one perfect fifth (7 semitones)**. Adjacent keys share 6 of 7 scale notes. This is why adjacent transitions sound seamless.

### All 24 Keys

**Minor (A ring):**

| Camelot | Key |
|---|---|
| 1A | A♭m |
| 2A | E♭m |
| 3A | B♭m |
| 4A | Fm |
| 5A | Cm |
| 6A | Gm |
| 7A | Dm |
| 8A | Am |
| 9A | Em |
| 10A | Bm |
| 11A | F#m |
| 12A | C#m |

**Major (B ring):**

| Camelot | Key |
|---|---|
| 1B | B |
| 2B | F#/G♭ |
| 3B | D♭ |
| 4B | A♭ |
| 5B | E♭ |
| 6B | B♭ |
| 7B | F |
| 8B | C |
| 9B | G |
| 10B | D |
| 11B | A |
| 12B | E |

### The Seven Camelot Moves (Scored)

| Move | Rule | Score | Description |
|---|---|---|---|
| **Same key** | `num_b == num_a`, `let_b == let_a` | 1.00 | Identical notes — zero harmonic change |
| **Adjacent** | `dist == 1`, same letter | 0.90 | One note changes — very safe, subtle lift/drop |
| **Parallel mode** | `num_b == num_a`, `let_b != let_a` | 0.85 | Same root, different mode — emotional shift without clash |
| **+7 energy boost** | `(num_a + 7) % 12 == num_b`, same letter | 0.80 | 1-semitone upward pitch shift — euphoric "key change" effect. Use sparingly (every 20–30 min), quick transitions only. Armin Van Buuren's signature move; mathematically equivalent to his "−5 move" |
| **Double step** | `dist == 2` | 0.65 | Slight tension — acceptable but use shorter overlap |
| **Distant** | `dist 3–5` | 0.30–0.50 | Significant harmonic clash — effects-only transitions, or echo-out reset |
| **Clash** | `dist ≥ 6` | 0.05 | Maximum dissonance — avoid except for deliberate dramatic breaks |

**Why clashing keys physically hurt:** When two notes from incompatible keys play simultaneously, their overtone series produce beating frequencies (amplitude fluctuations in the 10–50 Hz range) perceived as roughness. The cochlea groups nearby frequencies into critical bands (~1/3 octave wide). Two partials within the same critical band at ~25% of that bandwidth produce maximum roughness. Compatible keys share most notes, so their overtones coincide at identical frequencies rather than beating.

### Key Detection Pipeline

1. Compute CQT chromagram (log-frequency resolution matching musical pitch — preferred over STFT chroma for key detection)
2. Average across time → 12-dimensional pitch class distribution
3. Correlate against Krumhansl-Schmuckler profiles via Pearson correlation for all 24 key shifts
4. Select key with highest correlation; confidence = normalized correlation value

**K-S Major profile:** `[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]`
**K-S Minor profile:** `[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]`

**Limitation:** Single global key per track — misses modulations within tracks. DJs must always trust ears as final arbiter. Mixed In Key claims ~10% greater accuracy than competitors but still has meaningful error rates, especially on percussion-heavy material.

---

## 3. Compatibility Scoring Weights

Research-backed composite compatibility formula (SetFlow algorithm, validated against 1001Tracklists dataset of 1,557 mixes / 24,202 tracks):

```
compatibility = 0.35 × harmonic_match
              + 0.25 × beat_alignment
              + 0.15 × energy_smoothness
              + 0.15 × genre_proximity
              + 0.10 × timbral_similarity
              − vocal_clash_penalty
```

**Key empirical finding (Kim et al., ISMIR 2020, 1001Tracklists dataset):**
- 86.1% of tracks tempo-adjusted < 5% between transitions
- Only 2.5% of tracks key-transposed during mixing
- Transition lengths peak at multiples of 32 beats (phrase boundaries)
- 73.6% of cue point pairs across different DJs agree within 8 measures — confirming strong shared conventions

---

## 4. Beatmatching and Phrase Structure

### BPM Ranges by Genre

| Genre | BPM Range | Notes |
|---|---|---|
| Hip-hop | 80–100 | — |
| Trap | 60–75 | Half-time of 120–150 — mix at 1× or 2× |
| Deep house | 118–124 | — |
| House / Tech house | 124–130 | — |
| Techno | 125–140 | Hard techno: 140–150+ |
| Trance | 130–150 | — |
| Drum & bass | 165–180 | — |

### Double-Time / Half-Time Mixing

A track at X BPM aligns exactly with 2X or X/2 BPM. The slower track's beat 1 aligns with the faster track's beat 1; beat 3 aligns with beats 2 and 4. This enables cross-genre mixing (e.g., house at 128 BPM mixing with hip-hop at 64 BPM). Dubstep at 140 BPM has a half-time feel that lets DJs mix it with both 70 BPM and 140 BPM material.

### Phrase Structure

Electronic music hierarchy: **beats → bars (4 beats) → phrases (typically 8 bars / 32 beats) → sections**.

DJ-friendly tracks have 16–32 bar intros and outros designed for mixing. Music introduces audible changes at phrase boundaries (new instruments, drum fills, crashes, vocal entries) every 32 or 64 beats. When transitions align to these boundaries, the new track's elements coincide with expected change points — the transition feels intentional. Mixing off-phrase breaks the subconscious musical narrative.

**Best practice:** Always start/end transitions on 8-bar boundaries. 32-bar transitions are universally safe. 16-bar transitions are standard. 64-bar transitions work for minimal/techno.

### Time-Stretching Limits

Phase vocoder time-stretching introduces artifacts that become musically unacceptable beyond ±3–5 BPM of original tempo. Beyond this, use "repitch" mode (changes pitch proportionally to tempo, transparent but key-shifted) or BPM creep across multiple tracks.

---

## 5. Transition Techniques

### Genre-Specific Transition Duration

| Genre | Bars | Approximate Time (at typical BPM) |
|---|---|---|
| Techno (130 BPM) | 32–96 bars | 1–3 minutes |
| House (126 BPM) | 16–64 bars | 30 seconds – 2 minutes |
| Progressive / Trance (138 BPM) | 64–96 bars | 1–3+ minutes |
| Drum & bass (174 BPM) | 8–16 bars | ~17–33 seconds |
| Hip-hop (90 BPM) | 4–8 bars | ~10–20 seconds |
| Open format / Commercial | 4–16 bars | Emphasis on selection over blending |

### Long Blend (Workhorse Technique)

1. Start track B with bass EQ fully cut during a minimal section of track A
2. Gradually bring up B's fader over 16–64 bars
3. At a phrase boundary: simultaneously swap the bass (cut A's low EQ to zero, raise B's to neutral)
4. Fade out A's remaining mid/high frequencies

Duration: 30 seconds to 3+ minutes. Best for: minimal, deep house, ambient, progressive. Practitioners: Sasha, Digweed, Dixon, Tale of Us.

### Cut Mix (Hip-Hop / Electro)

Line up B's first beat at A's exact phrase boundary. Kill A and start B simultaneously on "the one" — no overlap. The audience subconsciously expects change at phrase boundaries, so the cut feels clean. Pioneered by Grandmaster Flash ("quick mix theory").

### Effects-Based Transitions

**Echo out:** Set echo to correct BPM subdivision, activate on last phrase, kill channel fader. Echo repeats fade naturally while new track enters underneath.

**Filter sweep:** High-pass filter on outgoing (progressively thinning) while low-pass on incoming (starting muffled, gradually revealing) — frequency handoff technique.

**Loop transition:** Set infinite loop on a clean section of A. Progressive shortening (8 → 4 → 2 → 1 → ½ → ¼ bars) creates a building "wash" effect that resolves with a cut to B.

---

## 6. EQ and Frequency Management

### The Bass Swap (Most Fundamental Technique)

Two simultaneous basslines cause: phase interference (constructive/destructive patterns), muddiness (ear cannot separate competing bass fundamentals), clipping (summed bass energy exceeds headroom), and acoustic overload on club subwoofer systems.

**Rule:** Only one dominant bass element occupies the low-frequency spectrum at any time.

- **Hard swap** (instant cut/boost at phrase boundary): Use when both tracks have strong, defined bass
- **Soft swap** (gradual over 8–16 bars): Use for deep house, minimal — prevents the swap from being audible

### Genre Frequency Profiles

| Genre | Kick Fundamental | Bass Role | Notes |
|---|---|---|---|
| House | 60–100 Hz | Bassline rides above kick at 100–200 Hz | Tight, punchy kick coexisting with warm melodic bass |
| Techno | 40–80 Hz | Aggressive, often same band as kick | Hard techno features very strong kick fundamentals |
| Hip-hop / 808 | Top kick: 100–200 Hz + click at 2–4 kHz | Sub-bass 30–60 Hz, long sustaining 808 tones | 808 tracks carry massive sub-bass energy — require specialized reproduction |
| Drum & bass / Dubstep | Punchier kicks (leave room for bass) | Bassline sub-bass role at 70–90 Hz | — |

**Practical implication:** When mixing between genres with different kick/bass frequency profiles, the bass swap crossover frequency should shift (house → techno swap at 120 Hz; 808 → house swap at 80 Hz to account for the 808's extended sub-bass).

### Gain Staging

Signal chain: software output → channel trim/gain → channel EQ → channel fader → effects → master output → amplifiers → speakers.

Clipping at any stage propagates distortion through the entire chain.

- **Trim (gain):** "Set and forget" — adjust so channel meters peak around −6 to −3 dBFS
- **Channel fader:** Artistic performance control — ideally near unity (~2/3 up)
- **Club rule:** Average in the green, peaks touching orange, never red

VU meters approximate perceived loudness (0 VU = −18 dBFS). Peak meters show instantaneous peaks relative to clipping.

---

## 7. ML Foundations for AI DJ Systems

### Audio Feature Extraction Pipeline

A complete AI DJ system requires these features per track:

| Feature | Method | Notes |
|---|---|---|
| BPM | `librosa.beat.beat_track`, Madmom RNN+DBN | Octave errors (70 vs 140) require dedicated classifier branch |
| Downbeat position | Madmom `DBNDownBeatTrackingProcessor` | Classifies beats as 1st/2nd/3rd/4th via Dynamic Bayesian Network |
| Key | CQT chromagram → Krumhansl-Schmuckler correlation | Or CNN classifiers for better accuracy |
| Energy | RMS via `librosa.feature.rms()`, LUFS via `pyloudnorm` | LUFS (ITU-R BS.1770) approximates human perceived loudness |
| Spectral centroid | Weighted mean of frequencies | Indicates brightness |
| Onset density | Detected onsets per unit time | Indicates complexity / activity |
| Vocal presence | Source separation → energy ratio | Or spectral proxy (300–3400 Hz vs <200 Hz ratio) |
| Danceability | Beat strength × tempo stability × bass ratio | Spotify-style composite |
| Timbral fingerprint | MFCCs (13–20 coefficients) | Capture spectral envelope shape |

### Beat Tracking State of the Art

Three generations:
1. **librosa:** Onset strength (spectral flux on mel-spectrogram) → autocorrelation → dynamic programming
2. **Madmom BLSTM:** 6 spectrogram representations → beat activation probabilities → Viterbi decoding through DBN
3. **Temporal Convolutional Networks (Davies & Böck, 2019):** 11 residual blocks with dilated convolutions, ~81.5-second receptive field, ELU activation — current non-Transformer state of the art

### Music Structure Segmentation

**Self-Similarity Matrices (SSMs):** Compute pairwise similarity between all frames using chroma or MFCC features. Homogeneous segments appear as diagonal blocks; repeated sections appear as off-diagonal stripes. Foote novelty detection convolves a checkerboard kernel along the SSM diagonal — peaks indicate section boundaries.

**All-In-One system (mir-aidj):** Source separation (4 stems) + Transformer trained on Harmonix Set → outputs BPM, beats, downbeats, and labeled segments.

### Transition Generation: DJtransGAN

Landmark paper (Chen et al., ICASSP 2022, Sony/Academia Sinica). GAN framework:
- **Generator:** Two DDSP components — differentiable equalizer + differentiable fader — predicts EQ and fader parameters over time to blend two tracks
- **Discriminator:** Judges whether result resembles real human DJ mixes
- **Training:** 284 crawled DJ mixes from Livetracklist

Listening test with 188 participants (46 experienced DJs) showed competitive results vs baselines. Main criticism: transitions sometimes too fast, especially vocal-to-vocal. Code open-sourced at `ChenPaulYu/DJtransGAN`.

**DDSP foundation (Engel et al., ICLR 2020, Google Magenta):** Integrates interpretable DSP modules (oscillators, filters, reverb) with deep learning by making them differentiable — enables end-to-end backpropagation through signal processing chains.

### Source Separation

**Demucs (Meta AI, Hybrid Transformer v4):** 9.00 dB SDR on MUSDB HQ, 4 stems (drums, bass, vocals, other). Industry implementations: Algoriddim djay's Neural Mix, VirtualDJ Stems 2.0, Serato Stems, Rekordbox Track Separation.

A 2025 Audio Developer Conference talk focused specifically on converting Demucs-class models to ONNX for real-time DJ software — this is the active frontier for latency-critical implementations.

### Sequence Planning as Constrained Optimization

DJ set ordering is analogous to the Traveling Salesman Problem — visit all tracks while minimizing transition "cost" — but with critical differences: the objective includes shaping an energy arc, diversity constraints prevent same-sounding consecutive tracks, and the cost function is asymmetric (A→B differs from B→A based on energy direction).

**Reward function components:**
```
R = w_h × harmonic_compatibility
  + w_e × energy_arc_fit
  + w_d × diversity_bonus
  + w_t × transition_quality
```

Where:
- `harmonic_compatibility` = penalize Camelot distance > 1
- `energy_arc_fit` = reward tracks matching the target energy curve
- `diversity_bonus` = penalize same artist/label/timbre in consecutive tracks
- `transition_quality` = bonus for aligned structural sections, non-overlapping vocals

### Five Unsolved Problems (as of 2024–2025)

1. **Crowd response modeling:** OpenPose body tracking is primitive. Mapping physical signals to abstract emotional states requires understanding far beyond current sensor data
2. **Creative track selection:** Current systems use similarity-based selection. Missing: deliberate contrast, tension-and-release narrative arcs, cultural context, thematic coherence
3. **Musically coherent transitions:** DJtransGAN achieves "competitive" results but transitions lack organic feel. Beat/downbeat tracking errors remain the primary bottleneck
4. **Long-term set planning:** No published system handles narrative arc planning across an entire set
5. **Genre convention understanding:** Human DJs' ability to deliberately break conventions — genre-bending, unexpected transitions — is essentially unaddressed

### Key Datasets

| Dataset | Content | Use |
|---|---|---|
| 1001Tracklists (Kim et al., ISMIR 2020) | 1,557 mixes, 24,202 tracks, metadata + boundary timestamps | Supervision for transition learning |
| UnmixDB (Schwarz & Fourer) | Auto-generated mixes with ground truth from CC-licensed tracks | Transition parameter estimation |
| DAFx 2022 DJ Mix Dataset | Labeled transition data with ground truth mixing parameters | Supervised transition learning |
| MUSDB18/HQ | 150 tracks, 5 stems | Source separation benchmark |
| FMA | 106,574 CC-licensed tracks, 161 genres | Genre classification |

---

## 8. Artist Reference: How Experts Implement These Principles

| Artist | Signature Technique | BPM Range | Style |
|---|---|---|---|
| **Ricardo Villalobos** | 5–10 min blends, loop/stretch in real-time, marathon sets 6–10 h | 120–128 | Micro-house, hypnotic |
| **Carl Cox** | 3–4 deck layering, live remixes via 2-bar loops on third deck, never plans sets | 128–140 | Techno/house |
| **Jeff Mills** | 70 records/hour, Roland TR-909 live drum programming as bridge | 140–150 | Detroit techno |
| **Sasha** | Ultra-long blends (32–64+ bars), progressive harmonic journeys | 125–132 | Progressive house |
| **Armin Van Buuren** | Camelot +7 "energy boost" move combined with energy level step-up | 130–140 | Trance |
| **Four Tet** | Ableton Live dual-laptop setup, 6+ h sets spanning techno/Motown/Indian raga | variable | Eclectic |
| **Amelie Lens** | Classic acid over big-room techno, flow state focus | 132–140 | Techno |

**Armin Van Buuren's documented +7 technique (Ultra 2017):**
"My Symphony Of You" (12A, Energy 7) → "Heading Up High" (7A, Energy 8). The Camelot jump of +7 produces a 1-semitone upward shift that registers psychologically as increased energy and brightness. Combined with a step-up in energy level — a "double technique" that creates a devastating crowd moment.

---

## 9. Practical Implementation Notes for AI RemixMate

### Camelot Move Selection Algorithm
```
1. Extract Camelot codes A and B
2. Compute clockwise distance: dist_cw = (num_b - num_a) % 12
3. Compute counter-clockwise distance: dist_ccw = (num_a - num_b) % 12
4. dist = min(dist_cw, dist_ccw)
5. Check for +7 boost: if dist_cw == 7 and let_a == let_b → "energy_boost" move
6. Apply scoring table (section 2 above)
7. Set recommended_transition_bars based on score and BPM:
   - score ≥ 0.85 and BPM 125-140 → 32-64 bars (techno long blend)
   - score ≥ 0.85 and BPM 118-130 → 16-32 bars (house blend)
   - score ≥ 0.85 and BPM 165+ → 8-16 bars (D&B short blend)
   - score < 0.4 → 4-8 bars (quick cut, minimize harmonic exposure)
```

### Bass Swap Timing
```
For transition_bars T:
  - Cut A's bass EQ: starts at bar T/2, reaches -∞ dB at bar T*0.75
  - B's bass EQ: starts at -∞ dB, reaches 0 dB at bar T*0.75
  - The swap happens in one moment at bar T*0.75 (hard swap)
  - For soft swap (deep house): linear ramp from bars T/3 to 2T/3
```

### Energy Level Interpretation
Mixed In Key's Energy Level 1–10 scale (not a volume metric):
- Levels 1–4: Ambient, minimal, breather tracks
- Level 5: Deep house, minimal — start of dancefloor energy
- Level 7: Guaranteed danceability
- Levels 8–9: Festival bangers
- Level 10: Extremely rare — maximum energy

For AI set planning: target average energy 5.0 for warm-up, 7.0 for peak time, step ±1 between consecutive tracks (Armin's documented pattern).

---

*Last updated: April 2026 | Based on: Pioneer DJ Research 2021, Kim et al. ISMIR 2020, Chen et al. ICASSP 2022, Krumhansl-Schmuckler 1990, SetFlow algorithm, 1001Tracklists dataset analysis*
