# AI RemixMate — Sophisticated Feature Spec + Codex Agent Build Plan

**Artifact purpose:** Convert professional DJ software concepts into an original, buildable roadmap for **AI RemixMate**.

**Primary source model:** djay Pro 2 manual concepts such as decks, waveform views, mixer, beatgrids, BPM/tempo, key lock, sync, slicer, slip mode, library analysis, Automix, Track Match, effects, loops, cue points, sampler, recording, MIDI mapping, and external mixing.

**Important boundary:** This is **not** a clone spec. Do not copy djay Pro UI, names, icons, layouts, text, proprietary effects, or branding. Use the manual only as a professional DJ workflow reference. AI RemixMate must remain an original AI-assisted remix, mashup, and transition system.

---

## 0. Product North Star

AI RemixMate should become an **AI-assisted remix engineering lab**, not merely a DJ player.

### Core positioning

> AI RemixMate analyzes songs, stems, beats, keys, energy curves, vocals, and timbre to help users discover compatible tracks, design DJ transitions, build mashups, and generate remix recipes.

### The app should answer

- What tracks mix well together?
- Where should Track B enter Track A?
- Should I mix by BPM, key, energy, vocal space, or phrase structure?
- Can these vocals sit over that instrumental?
- How do I make a 16-bar / 32-bar transition?
- Which stems should be active during each section?
- Can the app generate a draft mix timeline?

### Non-goals

- Do not build a generic music player.
- Do not build a visual clone of existing DJ apps.
- Do not depend on copyrighted audio samples for demos.
- Do not make real-time DSP perfect before basic workflows work.
- Do not claim “AI DJ” unless the system actually analyzes and makes useful musical decisions.

---

## 1. Foundational DJ Concepts Reframed for AI RemixMate

### 1.1 Deck

**DJ meaning:** A deck is an independent player for a track. It controls playback, tempo, cueing, waveform position, loops, and effects.

**AI RemixMate meaning:** A deck is a controllable audio/stem lane. It can hold a full track or a separated stem group.

#### Build requirements

Each deck must store:

```ts
export type DeckId = 'A' | 'B' | 'C' | 'D';

export interface DeckState {
  id: DeckId;
  trackId: string | null;
  loadedAssetUrl: string | null;
  stemMode: 'full' | 'vocals' | 'drums' | 'bass' | 'other' | 'multi-stem';
  isPlaying: boolean;
  isPaused: boolean;
  playheadSec: number;
  durationSec: number;
  bpm: number | null;
  detectedKey: string | null;
  effectiveBpm: number | null;
  pitchSemitone: number;
  keyLockEnabled: boolean;
  syncMode: 'off' | 'tempo' | 'beat';
  volume: number;      // 0..1
  gainDb: number;      // trim, e.g. -12..+12
  eq: {
    lowDb: number;
    midDb: number;
    highDb: number;
  };
  filter: {
    type: 'none' | 'lowpass' | 'highpass';
    amount: number;    // -1..1, negative = LPF, positive = HPF
  };
  loop: LoopState | null;
  cuePoints: CuePoint[];
  fxChain: EffectSlot[];
}
```

#### Agent notes

- **Frontend Agent:** build deck component without hard-coding two decks only. Use deck IDs.
- **Audio Engine Agent:** expose methods: `loadTrack`, `play`, `pause`, `seek`, `setVolume`, `setGain`, `setEQ`, `setFilter`, `setLoop`, `clearLoop`.
- **Backend Agent:** track metadata must be available before deck load when possible.
- **Test Agent:** verify deck state transitions: unloaded → loaded → playing → paused → seeking → unloaded.

---

## 2. Audio Engine Architecture

### 2.1 Recommended browser architecture

Use the **Web Audio API** as the lower-level audio graph.

#### Core graph

```text
Deck Source
  → Deck Gain Trim
  → EQ Low/Mid/High
  → Filter
  → FX Chain
  → Channel Volume Gain
  → Crossfader Gain
  → Master Gain
  → Destination / Recorder
```

#### Main node types

- `AudioContext`
- `AudioBufferSourceNode` or HTML media source wrapper
- `GainNode` for volume, gain, crossfader, master level
- `BiquadFilterNode` for EQ and filters
- `DelayNode` for echo/delay
- `ConvolverNode` for reverb later
- `MediaStreamAudioDestinationNode` for recording/export

### 2.2 Audio engine API

```ts
export interface RemixAudioEngine {
  init(): Promise<void>;
  loadDeck(deckId: DeckId, sourceUrl: string, metadata: TrackAnalysis): Promise<void>;
  unloadDeck(deckId: DeckId): void;
  play(deckId: DeckId, options?: PlayOptions): void;
  pause(deckId: DeckId): void;
  stop(deckId: DeckId): void;
  seek(deckId: DeckId, seconds: number): void;
  setDeckVolume(deckId: DeckId, value: number): void;
  setDeckGain(deckId: DeckId, gainDb: number): void;
  setCrossfader(position: number): void; // -1 left, 0 center, 1 right
  setMasterVolume(value: number): void;
  setEQ(deckId: DeckId, band: 'low' | 'mid' | 'high', db: number): void;
  setFilter(deckId: DeckId, amount: number): void;
  setPlaybackRate(deckId: DeckId, rate: number): void;
  startRecording(): Promise<void>;
  stopRecording(): Promise<Blob>;
}
```

### 2.3 Crossfader math

Avoid linear crossfade as the final version because perceived loudness may dip in the center. Start with linear for MVP, then implement equal-power crossfade.

```ts
function equalPowerCrossfade(position: number) {
  // position: -1 = deck A, 0 = equal, 1 = deck B
  const x = (position + 1) / 2; // 0..1
  const gainA = Math.cos(x * Math.PI / 2);
  const gainB = Math.cos((1 - x) * Math.PI / 2);
  return { gainA, gainB };
}
```

#### Acceptance criteria

- At far left: Deck A audible, Deck B muted.
- At center: both audible without obvious volume collapse.
- At far right: Deck B audible, Deck A muted.
- Crossfader automation supports 1, 2, 4, 8, 16, and 32 bar transitions.

---

## 3. Track Analysis Pipeline

### 3.1 Required metadata schema

```ts
export interface TrackAnalysis {
  trackId: string;
  title: string;
  artist?: string;
  sourcePath: string;
  durationSec: number;
  sampleRate: number;
  bpm: number;
  bpmConfidence?: number;
  beatTimesSec: number[];
  downbeatTimesSec: number[];
  barTimesSec: number[];
  phraseTimesSec: number[]; // 8/16/32-bar estimates
  key: MusicalKeyEstimate;
  energyCurve: TimeSeriesPoint[];
  spectralCentroidCurve?: TimeSeriesPoint[];
  mfccMean: number[];
  mfccVar: number[];
  chromaMean: number[];
  chromaCurve?: number[][];
  vocalActivityCurve?: TimeSeriesPoint[];
  sectionLabels?: TrackSection[];
  stems?: StemPaths;
  lyricsTranscript?: TranscriptSegment[];
  embeddingVector?: number[];
  analyzedAt: string;
  analyzerVersion: string;
}

export interface MusicalKeyEstimate {
  key: string;       // e.g. 'A minor'
  camelot?: string;  // e.g. '8A'
  confidence: number;
}
```

### 3.2 Backend analysis modules

Recommended Python modules:

```text
backend/audio_analysis/
  analyze_track.py
  bpm_detector.py
  beatgrid.py
  key_detector.py
  energy.py
  phrase_detector.py
  timbre_features.py
  compatibility.py
  stem_analysis.py
  transcript_analysis.py
```

### 3.3 Analysis stages

#### Stage A — ingest

- Validate file type.
- Normalize filename.
- Assign stable `trackId`.
- Store original source.
- Generate waveform peaks for frontend.

#### Stage B — rhythmic analysis

- Estimate BPM.
- Extract beat timestamps.
- Estimate downbeats.
- Build beatgrid.
- Create bar grid.
- Create phrase grid.

#### Stage C — harmonic analysis

- Compute chroma features.
- Estimate key.
- Convert key to Camelot-like compatibility notation internally.
- Store key confidence.

#### Stage D — timbre and texture

- Compute MFCC mean/variance.
- Spectral centroid / rolloff / bandwidth.
- Optional: percussion/vocal density.

#### Stage E — energy and structure

- RMS or loudness curve.
- Onset strength curve.
- Detect intro / verse / build / drop / breakdown / outro heuristically.
- Store transition candidates.

#### Stage F — stems

- Run Demucs or use existing separated stems.
- Store paths for vocals, drums, bass, other.
- Compute per-stem energy curves.
- Compute vocal density and drum density.

---

## 4. Beatgrid System

### 4.1 Concept

A beatgrid is a timeline of beat markers. It lets the app align playback, loops, effects, cue points, and transitions to musical time instead of raw seconds.

### 4.2 Data model

```ts
export interface BeatGrid {
  trackId: string;
  bpm: number;
  gridStartSec: number;
  beats: BeatMarker[];
  bars: BarMarker[];
  phrases: PhraseMarker[];
  confidence: number;
  manuallyEdited: boolean;
}

export interface BeatMarker {
  index: number;
  timeSec: number;
  beatInBar: 1 | 2 | 3 | 4;
  confidence?: number;
}

export interface BarMarker {
  index: number;
  startSec: number;
  beatIndex: number;
}

export interface PhraseMarker {
  index: number;
  startSec: number;
  bars: number;
  type?: 'intro' | 'verse' | 'build' | 'drop' | 'breakdown' | 'outro' | 'unknown';
}
```

### 4.3 Required functions

```ts
nearestBeat(timeSec: number): BeatMarker;
nextBeat(timeSec: number): BeatMarker;
nearestBar(timeSec: number): BarMarker;
nextBar(timeSec: number): BarMarker;
nextPhrase(timeSec: number, phraseBars?: number): PhraseMarker;
secondsToBeatIndex(timeSec: number): number;
beatIndexToSeconds(index: number): number;
quantizeTime(timeSec: number, unit: 'beat' | 'bar' | 'phrase'): number;
```

### 4.4 Manual correction UI

Build a simple Beatgrid Editor.

#### Controls

- Set grid start at current playhead.
- Nudge grid left/right by milliseconds.
- Double / halve BPM if detected wrong.
- Set BPM manually.
- Restore auto grid.
- Save manual override.

#### Codex prompt

> Implement a `BeatgridEditor` React component that receives `BeatGrid`, `DeckState`, and callbacks for `setGridStart`, `nudgeGrid`, `setBpm`, `halveBpm`, `doubleBpm`, `restoreAutoGrid`, and `saveManualGrid`. Render beat markers over the waveform. Keep it visually original and do not copy any third-party DJ app UI.

---

## 5. BPM, Tempo, Playback Rate, and Time Stretching

### 5.1 Definitions

- **BPM:** detected musical speed.
- **Tempo slider:** changes playback speed.
- **Playback rate:** browser-level speed multiplier.
- **Key lock:** keeps pitch stable while tempo changes.
- **Tempo sync:** match BPM between decks.
- **Beat sync:** match BPM and align beat/downbeat positions.

### 5.2 MVP implementation

For MVP, use `playbackRate`.

```ts
const targetRate = masterBpm / deckBpm;
```

Limit range:

```ts
const MIN_RATE = 0.75;
const MAX_RATE = 1.25;
```

### 5.3 Advanced implementation

PlaybackRate changes pitch. Later add a time-stretch/pitch-preserving engine.

Possible future paths:

- WebAssembly SoundTouch / Rubber Band style library.
- Server-side `librosa.effects.time_stretch` for offline preview generation.
- Python batch-rendered synced transition clips.
- AudioWorklet for real-time advanced processing.

### 5.4 Tempo sync algorithm

```ts
function tempoSync(deck: DeckState, master: DeckState) {
  if (!deck.bpm || !master.effectiveBpm) return;
  const rate = master.effectiveBpm / deck.bpm;
  return clamp(rate, 0.75, 1.25);
}
```

### 5.5 Beat sync algorithm

```text
1. Identify master deck beat at current master time.
2. Identify target deck nearest equivalent downbeat/bar.
3. Set target deck playbackRate = masterBpm / targetBpm.
4. Seek target deck to a quantized beat/bar start.
5. Start target deck on next master beat/bar boundary.
```

### 5.6 Acceptance criteria

- Tempo sync changes effective BPM display.
- Beat sync starts Deck B at a beat/bar boundary.
- A visual indicator shows sync mode.
- Users can disable sync and return to original tempo.

---

## 6. Key Detection, Key Lock, and Harmonic Mixing

### 6.1 What to build

AI RemixMate needs a harmonic compatibility system.

#### Track-level data

```ts
interface HarmonicProfile {
  estimatedKey: string;
  camelot: string;
  confidence: number;
  chromaMean: number[];
  compatibleKeys: string[];
}
```

### 6.2 Key compatibility scoring

```ts
export function harmonicCompatibility(a: HarmonicProfile, b: HarmonicProfile): number {
  // 1.0 = same key / relative-compatible
  // 0.8 = adjacent Camelot key
  // 0.6 = same root, major/minor switch
  // 0.3 = weak compatibility
  // 0.0 = likely clash
}
```

### 6.3 Key lock

MVP:

- Display key lock toggle.
- Do not claim high-quality pitch preservation unless implemented.
- For normal playbackRate, warn internally that pitch changes.

Advanced:

- Add pitch-shifting per semitone.
- Add key match: transpose Deck B to match Deck A or compatible key.
- Generate offline preview renders for key-shifted mashups.

### 6.4 AI feature

**Harmonic Match Assistant**

Outputs:

```json
{
  "sourceTrack": "Track A",
  "candidateTrack": "Track B",
  "keyA": "A minor",
  "keyB": "C major",
  "compatibility": 0.92,
  "reason": "Relative major/minor relationship. Low harmonic clash risk.",
  "recommendedAction": "Mix without pitch shift or try +0 semitones."
}
```

---

## 7. Waveform and Visual Timeline System

### 7.1 Required waveform layers

AI RemixMate should use layered waveforms:

```text
Layer 1: full waveform amplitude
Layer 2: beat markers
Layer 3: bar markers
Layer 4: phrase markers
Layer 5: cue points
Layer 6: loop region
Layer 7: energy curve
Layer 8: stem activity indicators
Layer 9: suggested transition windows
```

### 7.2 Frontend data contract

```ts
interface WaveformRenderData {
  trackId: string;
  peaks: number[];
  durationSec: number;
  beatTimesSec: number[];
  barTimesSec: number[];
  phraseMarkers: PhraseMarker[];
  cuePoints: CuePoint[];
  loop?: LoopState;
  energyCurve: TimeSeriesPoint[];
  transitionWindows: TransitionWindow[];
}
```

### 7.3 Components

```text
frontend/src/components/waveform/
  WaveformCanvas.tsx
  BeatMarkerLayer.tsx
  CuePointLayer.tsx
  LoopRegionLayer.tsx
  EnergyCurveLayer.tsx
  StemActivityLayer.tsx
  TransitionSuggestionLayer.tsx
```

### 7.4 Interaction requirements

- Click waveform to seek.
- Drag cue point to move.
- Drag loop boundary to resize loop.
- Hover marker to show metadata.
- Click AI suggestion to preview transition.
- Zoom in/out.
- Toggle full waveform vs stem waveform.

---

## 8. Mixer: Gain, EQ, Filter, Level Meter, Crossfader

### 8.1 Mixer data model

```ts
interface MixerState {
  masterVolume: number;
  crossfader: number; // -1..1
  channels: Record<DeckId, ChannelStrip>;
}

interface ChannelStrip {
  deckId: DeckId;
  volume: number;
  gainDb: number;
  eqLowDb: number;
  eqMidDb: number;
  eqHighDb: number;
  filterAmount: number;
  levelDb: number;
  isClipping: boolean;
  preCueEnabled: boolean;
}
```

### 8.2 EQ implementation

Use three filters per deck:

- Low shelf around ~100 Hz.
- Peaking mid around ~1 kHz.
- High shelf around ~8–10 kHz.

### 8.3 Filter knob

One knob:

- `0` = neutral.
- Negative = low-pass, progressively darker.
- Positive = high-pass, progressively thinner.

### 8.4 Avoid-the-red rule

Add clipping warnings.

```ts
if (levelDb > -1) showWarning('Clipping risk: lower gain or master volume');
```

### 8.5 AI mix coach

When two tracks play together, analyze likely clashes:

- Bass conflict: both low stems high.
- Vocal conflict: both vocal activity high.
- Harshness: both high-frequency energy high.
- Mud: low-mid buildup.

Output:

```json
{
  "warning": "Bass clash detected",
  "suggestion": "Cut Deck B low EQ for 8 bars, then swap bass at the drop.",
  "automation": [
    { "time": 0, "deck": "B", "eqLowDb": -12 },
    { "time": 16, "deck": "A", "eqLowDb": -12 },
    { "time": 16, "deck": "B", "eqLowDb": 0 }
  ]
}
```

---

## 9. Cue Points

### 9.1 Concept

Cue points are named positions in a track used for jumping, starting, or building transitions.

### 9.2 Data model

```ts
export interface CuePoint {
  id: string;
  trackId: string;
  timeSec: number;
  label: string;
  type: 'start' | 'intro' | 'verse' | 'chorus' | 'drop' | 'breakdown' | 'outro' | 'custom';
  color?: string;
  createdBy: 'user' | 'ai';
  confidence?: number;
}
```

### 9.3 AI cue suggestions

Heuristics:

- Intro start = first stable beat/bar.
- Drop = large energy rise after build.
- Breakdown = energy drop with lower drums.
- Vocal start = first sustained vocal activity.
- Outro = last phrase with reduced energy.

### 9.4 Features

- Set cue at playhead.
- Jump to cue.
- Rename cue.
- Delete cue.
- Snap cue to nearest beat/bar.
- Generate AI cues.
- Export cue map.

### 9.5 Acceptance criteria

- Cue points persist per track.
- Cue jump is sample-accurate enough for user perception.
- AI cues are visually marked as suggestions, not facts.
- Users can approve/reject AI cues.

---

## 10. Loops

### 10.1 Concept

A loop repeats a region of audio. In DJ mixing, loops are usually beat-aligned so the rhythm stays stable.

### 10.2 Data model

```ts
export interface LoopState {
  deckId: DeckId;
  enabled: boolean;
  startSec: number;
  endSec: number;
  lengthBeats: number;
  quantized: boolean;
  createdBy: 'user' | 'ai';
}
```

### 10.3 Loop sizes

Support:

```text
1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32 beats
```

### 10.4 Loop operations

```ts
setAutoLoop(deckId, beats);
setLoopIn(deckId);
setLoopOut(deckId);
halveLoop(deckId);
doubleLoop(deckId);
clearLoop(deckId);
moveLoop(deckId, beatOffset);
```

### 10.5 Remix intelligence

AI should suggest loopable regions:

- Clean drum intro.
- Instrumental outro.
- Vocal phrase.
- Build-up segment.
- One-bar groove.
- Drop hook.

Output:

```json
{
  "loopCandidate": {
    "startSec": 42.12,
    "endSec": 57.48,
    "lengthBeats": 32,
    "reason": "Stable drums, low vocal activity, strong transition bed."
  }
}
```

---

## 11. Slicer Mode

### 11.1 Concept

Slicer divides the current 4- or 8-beat region into beat slices. The user can retrigger slices like temporary hot cues.

### 11.2 AI RemixMate version

Build **Beat Slicer** for remix performance and sample discovery.

```ts
interface SlicePad {
  index: number;
  startSec: number;
  endSec: number;
  beatIndex: number;
  label: string;
}
```

### 11.3 Operations

- Generate 4-slice mode.
- Generate 8-slice mode.
- Trigger slice pad.
- Quantize slice trigger.
- Record slicer performance as automation.

### 11.4 Use cases

- Vocal chop preview.
- Drum fill creation.
- Build-up stutter.
- Live remixing.

### 11.5 Implementation warning

Browser scheduling must be tight. Use `AudioContext.currentTime` scheduling rather than React render timing.

---

## 12. Slip Mode

### 12.1 Concept

Slip mode means the audible playhead can be manipulated temporarily while the hidden timeline keeps moving. When the manipulation ends, playback resumes where the song would have been.

### 12.2 Data model

```ts
interface SlipState {
  enabled: boolean;
  slipStartedAtAudioContextTime: number | null;
  originalDeckTimeSec: number | null;
  hiddenPlayheadSec: number | null;
  audibleOverride: 'scratch' | 'loop' | 'slice' | 'cue' | null;
}
```

### 12.3 MVP

Implement slip loop first:

```text
1. User enables slip.
2. User triggers a loop/slice.
3. Hidden playhead continues advancing.
4. User exits loop/slice.
5. Deck seeks to hidden playhead and continues.
```

### 12.4 Acceptance criteria

- Slip loop does not permanently move track position.
- Visual UI shows hidden playhead ghost marker.
- Slip mode can be disabled instantly.

---

## 13. Effects System

### 13.1 Effect architecture

```ts
export interface EffectSlot {
  id: string;
  type: EffectType;
  enabled: boolean;
  wet: number;
  params: Record<string, number>;
}

type EffectType =
  | 'filter'
  | 'echo'
  | 'delay'
  | 'reverb'
  | 'bitcrusher'
  | 'gate'
  | 'stutter'
  | 'phaser'
  | 'flanger';
```

### 13.2 MVP effects

Build in this order:

1. Filter
2. EQ
3. Echo/delay
4. Gate
5. Stutter
6. Reverb
7. Bitcrusher

### 13.3 Effect controls

Each effect needs:

- Enable/disable.
- Wet/dry amount.
- Main parameter.
- Beat-sync option where relevant.
- Reset to default.

### 13.4 AI effect recipes

Instead of random effect controls, create useful transition recipes:

```json
{
  "name": "Echo Out Vocal Into Drop",
  "bars": 8,
  "steps": [
    { "bar": 1, "deck": "A", "effect": "echo", "wet": 0.2 },
    { "bar": 5, "deck": "A", "effect": "echo", "wet": 0.55 },
    { "bar": 7, "deck": "A", "filter": "highpass", "amount": 0.4 },
    { "bar": 8, "deck": "B", "volume": 1.0 }
  ]
}
```

### 13.5 Agent notes

- **Audio Agent:** effect chain must be reorderable later.
- **Frontend Agent:** design effect rack as original UI.
- **AI Agent:** generate transition recipes, not low-level DSP code at first.
- **Test Agent:** snapshot effect state and verify toggles do not break audio graph.

---

## 14. Sampler

### 14.1 Concept

Sampler triggers short audio clips independently from deck playback.

### 14.2 AI RemixMate version

Sampler should support:

- Imported samples.
- Recorded snippets from deck.
- AI-suggested vocal chops.
- One-shot drums.
- Transition risers/impacts from user-owned files.

### 14.3 Data model

```ts
interface SamplePad {
  id: string;
  name: string;
  sourceUrl: string;
  durationSec: number;
  mode: 'one-shot' | 'hold' | 'loop';
  quantize: 'off' | 'beat' | 'bar';
  volume: number;
  createdFromTrackId?: string;
  sourceStartSec?: number;
  sourceEndSec?: number;
}
```

### 14.4 Features

- 12 sample pads.
- Trigger sample.
- Stop sample.
- Record from deck selection.
- Slice selected loop into pads.
- Quantized pad launch.
- Save sample bank.

---

## 15. Library Atlas

### 15.1 Concept

A DJ library is not just file browsing. It is preparation infrastructure.

### 15.2 AI RemixMate library features

- Import files/folders.
- Batch analyze songs.
- Search metadata.
- Filter by BPM range.
- Filter by key.
- Filter by energy.
- Filter by vocal/instrumental density.
- Filter by stem availability.
- Sort by mix compatibility.
- Preview track.
- Add to queue.
- Add to set.

### 15.3 Track table columns

```text
Title | Artist | BPM | Key | Energy | Vocal % | Stem Status | Best Matches | Last Analyzed | Tags
```

### 15.4 Backend endpoints

```http
POST /api/library/import
POST /api/library/analyze
GET  /api/library/tracks
GET  /api/library/tracks/:trackId
GET  /api/library/tracks/:trackId/waveform
GET  /api/library/tracks/:trackId/matches
PATCH /api/library/tracks/:trackId/tags
```

### 15.5 Batch analysis queue

```ts
interface AnalysisJob {
  id: string;
  trackId: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  stage: 'ingest' | 'waveform' | 'bpm' | 'key' | 'stems' | 'embeddings';
  progress: number;
  error?: string;
}
```

---

## 16. Track Match / Compatibility Engine

### 16.1 Difference between similarity and mix compatibility

**Similarity** asks: do these songs sound alike?

**Mix compatibility** asks: can these songs transition, layer, or mash up well?

They are related but not identical.

### 16.2 Compatibility formula

```ts
interface CompatibilityScore {
  sourceTrackId: string;
  candidateTrackId: string;
  total: number;
  bpm: number;
  key: number;
  energy: number;
  phrase: number;
  timbre: number;
  vocalClash: number;
  stemCompatibility: number;
  explanation: string[];
  recommendedTransition: TransitionSuggestion;
}
```

Suggested weighting:

```text
BPM compatibility          20%
Harmonic/key compatibility 20%
Energy curve compatibility 15%
Phrase alignment           15%
Timbre similarity          10%
Vocal clash avoidance      10%
Stem compatibility         10%
```

### 16.3 BPM score

```ts
function bpmCompatibility(a: number, b: number): number {
  const ratio = Math.max(a, b) / Math.min(a, b);
  const altRatio = Math.min(
    Math.abs(a - b),
    Math.abs(a - b * 2),
    Math.abs(a * 2 - b)
  );
  // Convert delta to 0..1 score.
}
```

### 16.4 Vocal clash score

High score means low clash.

```text
If Track A vocal activity high during transition and Track B vocal activity high, penalize.
If Track B instrumental intro and Track A vocal outro, reward.
If one track has isolated vocal stem and other has instrumental stem, reward for mashup.
```

### 16.5 Output example

```json
{
  "total": 0.87,
  "bpm": 0.95,
  "key": 0.90,
  "energy": 0.82,
  "phrase": 0.88,
  "timbre": 0.74,
  "vocalClash": 0.91,
  "stemCompatibility": 0.86,
  "explanation": [
    "BPM difference is small enough for clean tempo sync.",
    "Keys are harmonically compatible.",
    "Track B has a low-vocal intro suitable under Track A outro.",
    "Energy rises in Track B near a phrase boundary, suitable for a 16-bar transition."
  ]
}
```

---

## 17. Automix / Set Builder

### 17.1 Concept

Automix sequences tracks and applies transitions automatically.

### 17.2 AI RemixMate version

Build **Set Builder** as a planning system, not just autoplayer.

#### Inputs

- Seed track.
- Desired length.
- Energy arc: chill → build → peak → cooldown.
- BPM range.
- Key strictness.
- Vocal clash strictness.
- Transition length preference.

#### Outputs

- Ordered setlist.
- Transition points.
- Transition recipes.
- Cue points.
- Loop recommendations.
- Stem automation plan.

### 17.3 Set model

```ts
interface MixSet {
  id: string;
  name: string;
  tracks: SetTrack[];
  transitions: TransitionPlan[];
  createdBy: 'user' | 'ai';
}

interface SetTrack {
  trackId: string;
  order: number;
  startAtSec?: number;
  endAtSec?: number;
  role: 'opener' | 'builder' | 'peak' | 'bridge' | 'closer';
}
```

### 17.4 Transition plan model

```ts
interface TransitionPlan {
  id: string;
  fromTrackId: string;
  toTrackId: string;
  startTimeA: number;
  startTimeB: number;
  durationBars: number;
  strategy: 'blend' | 'bass-swap' | 'echo-out' | 'filter-sweep' | 'drop-swap' | 'vocal-over-instrumental';
  automation: AutomationEvent[];
  explanation: string;
}
```

---

## 18. Remix Lab: Stem-Aware Remixing

### 18.1 Stems

AI RemixMate already has a natural advantage because it uses source separation.

Supported stems:

```text
vocals
bass
drums
other
full mix
```

### 18.2 Stem mixing features

- Solo/mute stems.
- Crossfade full tracks.
- Crossfade stems independently.
- Vocal over instrumental preview.
- Drum swap.
- Bass swap.
- Drop reconstruction.
- Stem-level EQ.
- Stem-level effects.

### 18.3 Stem compatibility score

```text
Vocal A + Instrumental B:
- key compatibility
- BPM compatibility
- vocal phrase alignment
- instrumental density below vocal range
- energy contour fit
```

### 18.4 Remix recipe generator

Example output:

```md
## Remix Recipe: Track A vocals over Track B instrumental

1. Use Track B instrumental from 0:32 as the bed.
2. Bring Track A vocals at 0:48, snapped to bar 17.
3. High-pass Track A vocal below 120 Hz.
4. Cut Track B other stem by -3 dB during vocal phrases.
5. Add 1/4 echo to final vocal word before Track B drop.
6. Bring Track B drums fully at bar 33.
```

---

## 19. Recording and Export

### 19.1 MVP recording

Use browser recording from the master output.

Features:

- Start recording.
- Stop recording.
- Save blob.
- Download/export audio.
- Store recording metadata.

### 19.2 Offline render later

For higher quality, create server-side mix rendering:

```text
Input: tracks, stems, transition plan, automation events
Output: WAV/MP3 preview render
```

### 19.3 Export data model

```ts
interface MixRecording {
  id: string;
  setId?: string;
  createdAt: string;
  durationSec: number;
  audioUrl: string;
  format: 'webm' | 'wav' | 'mp3';
  transitionPlanIds: string[];
}
```

---

## 20. Shortcuts and Agentic Controls

### 20.1 Keyboard shortcuts

Build shortcuts around action names, not key names.

```ts
interface ShortcutBinding {
  action: string;
  keys: string[];
  scope: 'global' | 'deck' | 'library' | 'sampler' | 'waveform';
}
```

### 20.2 Suggested actions

```text
Deck A play/pause
Deck B play/pause
Set cue
Jump cue
Loop on/off
Loop halve/double
Crossfade left/center/right
Nudge deck forward/back
Sync deck
Load selected to Deck A/B
Add selected to set
Trigger sample pad 1-12
```

### 20.3 AI command palette

Add natural language actions:

- “Find a good next track.”
- “Make this transition smoother.”
- “Suggest cue points.”
- “Loop the clean intro.”
- “Build a 20-minute high-energy set.”
- “Find vocals that fit this instrumental.”

---

## 21. API Design

### 21.1 Track analysis

```http
POST /api/analyze/track
POST /api/analyze/batch
GET  /api/analyze/jobs/:jobId
```

### 21.2 Matching

```http
GET /api/match/:trackId
POST /api/match/compare
POST /api/match/set-builder
```

### 21.3 Remix planning

```http
POST /api/remix/transition-plan
POST /api/remix/stem-mashup
POST /api/remix/recipe
POST /api/remix/render-preview
```

### 21.4 Cue/loop persistence

```http
GET    /api/tracks/:trackId/cues
POST   /api/tracks/:trackId/cues
PATCH  /api/tracks/:trackId/cues/:cueId
DELETE /api/tracks/:trackId/cues/:cueId

GET    /api/tracks/:trackId/loops
POST   /api/tracks/:trackId/loops
DELETE /api/tracks/:trackId/loops/:loopId
```

---

## 22. Database Schema Draft

### 22.1 Tables

```sql
tracks(
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  artist TEXT,
  source_path TEXT NOT NULL,
  duration_sec REAL,
  bpm REAL,
  key_name TEXT,
  camelot_key TEXT,
  energy_mean REAL,
  vocal_density REAL,
  analyzed_at TEXT,
  analyzer_version TEXT
);

track_features(
  track_id TEXT PRIMARY KEY,
  mfcc_mean_json TEXT,
  mfcc_var_json TEXT,
  chroma_mean_json TEXT,
  embedding_json TEXT,
  FOREIGN KEY(track_id) REFERENCES tracks(id)
);

beatgrids(
  track_id TEXT PRIMARY KEY,
  bpm REAL,
  grid_start_sec REAL,
  beat_times_json TEXT,
  bar_times_json TEXT,
  phrase_times_json TEXT,
  confidence REAL,
  manually_edited INTEGER,
  FOREIGN KEY(track_id) REFERENCES tracks(id)
);

cue_points(
  id TEXT PRIMARY KEY,
  track_id TEXT,
  time_sec REAL,
  label TEXT,
  type TEXT,
  created_by TEXT,
  confidence REAL,
  FOREIGN KEY(track_id) REFERENCES tracks(id)
);

stems(
  track_id TEXT PRIMARY KEY,
  vocals_path TEXT,
  drums_path TEXT,
  bass_path TEXT,
  other_path TEXT,
  model_name TEXT,
  created_at TEXT,
  FOREIGN KEY(track_id) REFERENCES tracks(id)
);

compatibility_scores(
  source_track_id TEXT,
  candidate_track_id TEXT,
  total REAL,
  bpm REAL,
  key_score REAL,
  energy REAL,
  phrase REAL,
  timbre REAL,
  vocal_clash REAL,
  stem_score REAL,
  explanation_json TEXT,
  PRIMARY KEY(source_track_id, candidate_track_id)
);
```

---

## 23. Testing Strategy

### 23.1 Unit tests

- BPM compatibility score.
- Harmonic compatibility score.
- Crossfader gain math.
- Beat quantization.
- Loop halve/double.
- Cue persistence.
- Track analysis schema validation.

### 23.2 Integration tests

- Import track → analyze → view in library.
- Load Deck A/B → play/pause → crossfade.
- Generate match list → inspect explanation.
- Generate transition plan → render timeline.
- Save cue → reload app → cue remains.

### 23.3 Audio regression tests

- Analysis output should remain stable for fixture audio.
- BPM tolerance: ±2 BPM for test fixture.
- Beatgrid first beat within tolerance for fixture.
- No clipping above 0 dBFS in rendered preview.

### 23.4 UI smoke tests

- Library page loads.
- Deck page loads.
- Track can be selected.
- Waveform renders placeholder or peaks.
- Shortcuts modal opens.
- Empty states show CTA.

---

## 24. Codex Agent Operating Plan

Use focused agents. Do not ask one agent to build the entire product at once.

### Agent 1 — Architecture Auditor

**Mission:** Inspect existing AI RemixMate repo and map current files to this spec.

**Prompt:**

```text
You are the Architecture Auditor for AI RemixMate. Inspect the repository and produce a concise map of existing frontend, backend, audio analysis, and test modules. Identify what already exists, what is partially implemented, what is missing, and what is risky. Do not make code changes. Pay special attention to tracked node_modules, large generated files, stale build artifacts, broken imports, and uncommitted UI work. Output a markdown report with file paths and next recommended patches.
```

### Agent 2 — Audio Engine Builder

**Mission:** Build the browser audio engine abstraction.

**Prompt:**

```text
You are the Audio Engine Builder for AI RemixMate. Implement a Web Audio API based engine abstraction with deck loading, play, pause, seek, volume, gain, EQ, filter, master volume, and equal-power crossfader. Keep the implementation framework-agnostic where possible. Add TypeScript types and unit tests for crossfader math and state transitions. Do not redesign the UI. Do not add external dependencies unless justified.
```

### Agent 3 — Track Analysis Backend

**Mission:** Create Python analysis pipeline.

**Prompt:**

```text
You are the Track Analysis Backend Agent. Implement or refactor the Python audio analysis pipeline to output a stable TrackAnalysis JSON schema. Include duration, sample rate, BPM, beat times, rough bar/phrase times, chroma features, MFCC mean/variance, energy curve, and analyzer version. Use existing project conventions. Add CLI command: analyze-track <path> --out <json>. Include tests using a small fixture or mocked arrays if real audio fixtures are unavailable.
```

### Agent 4 — Beatgrid and Quantization Agent

**Mission:** Implement beatgrid data structures and time snapping.

**Prompt:**

```text
You are the Beatgrid Agent. Add BeatGrid types and pure utility functions: nearestBeat, nextBeat, nearestBar, nextBar, beatIndexToSeconds, secondsToBeatIndex, and quantizeTime. Build unit tests for edge cases: before first beat, after last beat, exact beat, between beats, and missing beatgrid. Do not touch audio playback yet.
```

### Agent 5 — Waveform UI Agent

**Mission:** Create waveform visualization layers.

**Prompt:**

```text
You are the Waveform UI Agent. Build an original waveform component that can render peaks plus overlays for beats, bars, cue points, loop regions, energy curve, and transition suggestions. It must accept data props and callbacks for seek, set cue, move cue, and select transition suggestion. Use existing styling conventions. Do not copy the layout of any third-party DJ software.
```

### Agent 6 — Mixer UI Agent

**Mission:** Build mixer controls.

**Prompt:**

```text
You are the Mixer UI Agent. Implement mixer components for channel volume, gain trim, low/mid/high EQ, filter knob, level meter placeholder, master volume, and crossfader. Wire them to existing or new state hooks but keep audio engine integration behind callbacks. Add accessible labels and keyboard operability.
```

### Agent 7 — Compatibility Engine Agent

**Mission:** Build match scoring.

**Prompt:**

```text
You are the Compatibility Engine Agent. Implement mix compatibility scoring separate from raw similarity. Score BPM fit, key fit, energy fit, phrase fit, timbre similarity, vocal clash avoidance, and stem compatibility. Return a total score plus explanations. Add tests with synthetic metadata pairs. Do not require real audio files for tests.
```

### Agent 8 — Cue and Loop Agent

**Mission:** Implement cue/loop state and persistence.

**Prompt:**

```text
You are the Cue and Loop Agent. Add cue point and loop data models, reducers/actions, API persistence if backend exists, and UI controls to set/jump/delete cues and set/halve/double/clear loops. Loops should snap to beatgrid when available. Add tests for reducer logic and quantized loop boundaries.
```

### Agent 9 — AI Remix Planner Agent

**Mission:** Generate structured transition/remix plans.

**Prompt:**

```text
You are the AI Remix Planner Agent. Implement deterministic first-pass transition planning from track analysis and compatibility scores. The planner should suggest transition duration, entry point, exit point, strategy, EQ/filter automation, and plain-English explanation. Do not call an LLM yet. Make outputs JSON-serializable and testable.
```

### Agent 10 — Smoke Test and Hardening Agent

**Mission:** Verify app runs and basic flows work.

**Prompt:**

```text
You are the Smoke Test and Hardening Agent. Add or repair smoke tests for loading the app, opening Library/Mix Deck/Set Builder, rendering empty states, selecting a track fixture, opening shortcuts, and verifying no console-breaking import errors. Do not make broad visual redesigns. Keep patches small and reversible.
```

---

## 25. Implementation Roadmap

### Phase 0 — Repo cleanup

- Remove tracked `node_modules` if present.
- Confirm package manager.
- Confirm frontend build works.
- Confirm backend analysis scripts run.
- Add `.gitignore` rules.
- Add fixture strategy.

### Phase 1 — Core analysis + library

- Stable `TrackAnalysis` schema.
- Batch analysis job queue.
- Library table with BPM/key/energy/stem status.
- Track detail view.
- Waveform peaks.

### Phase 2 — Deck and mixer MVP

- Two decks.
- Load tracks.
- Play/pause/seek.
- Volume and crossfader.
- Master volume.
- Basic waveform.

### Phase 3 — Beat-aware features

- Beatgrid.
- Quantized cue points.
- Loops.
- Tempo sync.
- Beat sync MVP.
- Transition suggestion markers.

### Phase 4 — AI compatibility and set building

- Match scoring.
- Match explanations.
- Set builder.
- Transition planner.
- Energy arc planning.

### Phase 5 — Stem remixing

- Stem deck mode.
- Vocal/instrumental compatibility.
- Stem mute/solo.
- Stem-level preview.
- Remix recipe generator.

### Phase 6 — Export and polish

- Recording.
- Offline render preview.
- Saved mix vault.
- Automation editor.
- Better tests.
- Demo data and README.

---

## 26. File/Folder Plan

```text
frontend/src/audio/
  RemixAudioEngine.ts
  crossfader.ts
  filters.ts
  effects.ts
  recorder.ts

frontend/src/features/decks/
  Deck.tsx
  DeckControls.tsx
  deckTypes.ts
  useDeckState.ts

frontend/src/features/mixer/
  Mixer.tsx
  ChannelStrip.tsx
  EQControls.tsx
  Crossfader.tsx
  LevelMeter.tsx

frontend/src/features/waveform/
  WaveformCanvas.tsx
  layers/
    BeatLayer.tsx
    CueLayer.tsx
    LoopLayer.tsx
    EnergyLayer.tsx
    TransitionLayer.tsx

frontend/src/features/library/
  LibraryAtlas.tsx
  TrackTable.tsx
  TrackFilters.tsx
  MatchPanel.tsx

frontend/src/features/remix/
  RemixLab.tsx
  TransitionPlanner.tsx
  StemMixer.tsx
  RemixRecipePanel.tsx

shared/types/
  trackAnalysis.ts
  beatgrid.ts
  deck.ts
  mixer.ts
  compatibility.ts
  remixPlan.ts

backend/audio_analysis/
  analyze_track.py
  beatgrid.py
  features.py
  key_detection.py
  compatibility.py
  stems.py

backend/api/
  library_routes.py
  analysis_routes.py
  match_routes.py
  remix_routes.py

tests/
  frontend/
  backend/
  fixtures/
```

---

## 27. Quality Bar

A feature is not done unless it has:

- Clear type/schema.
- UI state.
- Backend or mock data path.
- Basic tests.
- Empty/loading/error states.
- No copied third-party UI.
- No hard-coded song-specific logic.
- No claims that exceed implementation.

---

## 28. Immediate Next Codex Task

Run this first:

```text
Inspect the current AI RemixMate repository and create a repository gap report against the AI_REMIXMATE_CODEX_FEATURE_SPEC.md artifact. Do not change code. Identify existing files/components that map to: Library Atlas, Mix Deck, Waveform Deck, Set Builder, Mix Vault, Operations, audio analysis, compatibility engine, Demucs stems, Whisper lyrics, and smoke tests. Flag merge hygiene problems such as tracked node_modules or generated assets. End with a prioritized 10-patch implementation plan.
```

Then run this second:

```text
Implement the shared TypeScript types for TrackAnalysis, BeatGrid, DeckState, MixerState, CuePoint, LoopState, CompatibilityScore, and TransitionPlan. Add pure utility functions for crossfader gain math and beat quantization. Add unit tests. Do not touch UI yet.
```

---

## 29. Reference Notes

This spec is informed by common DJ software concepts including decks, mixer, crossfader, waveform, BPM/tempo, beatgrid, key lock, sync, cue points, loops, slicer, slip mode, sampler, effects, library analysis, Automix, Track Match, and recording. Implementation should use original AI RemixMate product language and design.

Technical references used while drafting:

- Web Audio API for browser audio graph concepts.
- MDN GainNode and BiquadFilterNode documentation for volume and EQ/filter primitives.
- librosa documentation for beat tracking, tempo, chroma, MFCC, and feature extraction.
- WaveSurfer.js documentation for interactive waveform visualization.
- Demucs / Hybrid Demucs references for source separation into vocals, drums, bass, and other stems.
