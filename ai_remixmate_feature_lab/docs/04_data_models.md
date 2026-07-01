# Data Models

The models below are the shared language for schemas, TypeScript interfaces, Python dataclasses, UI props, and future API payloads.

## TrackAnalysis

Describes one analyzed track: identity, duration, BPM, key, beatgrid, energy curve, timbre, vocal activity, sections, cues, stems, and confidence. It should be JSON-serializable and stable enough to persist.

```ts
const track: TrackAnalysis = {
  id: "track-a",
  title: "Track A",
  durationSec: 362.5,
  bpm: 128,
  key: "A minor",
  camelot: "8A",
  beatgrid: { bpm: 128, beatTimes: [0, 0.469], downbeats: [0], beatsPerBar: 4, confidence: 0.92 },
  energyCurve: [0.2, 0.4, 0.7],
  timbreVector: [0.1, 0.2, 0.3],
  vocalActivity: [0, 0.1, 0.8],
  sections: [{ label: "intro", startSec: 0, endSec: 32, confidence: 0.7 }]
}
```

## BeatGrid

Holds positive BPM, ascending beat timestamps, downbeats, beats per bar, and confidence. Beat times must be ascending and downbeats should be a subset of the musical grid.

## DeckState

Represents one deck: loaded track ID, transport state, current time, tempo percent, pitch semitones, key lock, sync, selected cue, and loop state.

## MixerState

Represents crossfader position, master gain, cue mix, limiter status, and channel strips with gain, volume, EQ, filter, mute, solo, and meter values.

## CuePoint

A named timestamp with type (`memory`, `hot`, `load`, `mix_in`, `mix_out`, `drop`, `vocal`), color, optional beat index, and confidence.

## LoopState

A loop window with non-negative start, end greater than start, length in beats, active state, and optional quantize value.

## EffectState

An enabled/disabled effect with wet amount, type, and numeric parameters. It validates state but does not implement DSP.

## StemManifest

Lists stem files for a track, including stem type, path, duration, sample rate, channel count, and optional quality fields.

## CompatibilityScore

Composite and component scores in the 0-1 range plus explanations and warnings. Vocal clash is represented as a penalty/risk.

## TransitionPlan

Entry and exit times, transition length in bars, cue suggestions, loop suggestions, EQ/filter automation notes, score summary, and warnings.

## RemixRecipe

A human-readable ordered plan with steps, stem guidance, timing guidance, transition method, risks, and export metadata.

## AutomationLane

A target parameter and sorted timestamped automation points with interpolation mode.

## ExportManifest

A reproducible export document containing source tracks, recipe, optional rendered audio files, schema version, created timestamp, and warnings.

## TrainingEvent

An explicit supervised learning event for the neural layer. It contains a feature model name, bounded input vector, bounded target vector, source, timestamp, optional weight, and metadata. It is the only supported way for the lab's neural models to learn.

```ts
const event: TrainingEvent = {
  id: "rating-001",
  featureName: "compatibility_score",
  inputVector: [0.64, 0.4, 0.66],
  targetVector: [0.9],
  source: "user_feedback",
  createdAt: "2026-06-30T23:00:00Z"
}
```

## NeuralFeatureModel

Metadata and JSON-serializable weights for one tiny online MLP. Fields include input size, hidden size, output size, learning rate, version, examples seen, last loss, and weight matrices.

## ModelRegistry

A JSON state object containing all adaptive feature models. It lets future backend jobs persist neural state without touching source code or root project files.
