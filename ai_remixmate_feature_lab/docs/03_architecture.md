# Architecture

The feature lab is layered so each piece can be tested before integration. Schemas define contracts, TypeScript and Python models mirror them, pure intelligence modules operate on those models, prototypes render typed state, and IO helpers export reproducible JSON plans.

## Layers

- Audio analysis layer: optional wrappers for BPM, beatgrid, chroma key, MFCC, energy, sections, stems, and waveform summaries. Heavy dependencies are imported inside functions.
- Feature store layer: JSON-serializable `TrackAnalysis`, `StemManifest`, and `ExportManifest` objects that can later map to SQLite or existing metadata files.
- Compatibility intelligence layer: deterministic BPM, key, energy, timbre, and vocal clash scoring combined into a transparent score.
- Audio playback layer: deliberately not implemented here; primitives expose math that future Web Audio or Python renderers can consume.
- Deck state layer: `DeckState` captures loaded track, transport, tempo, key lock, cue, loop, and sync settings.
- Mixer state layer: `MixerState` captures channels, EQ, filters, crossfader, and master gain.
- Waveform visualization layer: prototype components render peaks, playhead, cues, beat/downbeat hints, and loop ranges.
- Remix planning layer: transition planner and recipe planner produce actionable steps with warnings.
- Export layer: manifests and JSON helpers preserve plan, source metadata, generated files, and validation state.

## Full System Data Flow

```mermaid
flowchart LR
  A["Audio file or library track"] --> B["Optional analysis wrappers"]
  B --> C["TrackAnalysis JSON"]
  C --> D["Compatibility intelligence"]
  C --> E["Deck and waveform prototypes"]
  D --> F["Transition planner"]
  F --> G["Remix recipe planner"]
  G --> H["ExportManifest JSON"]
  H --> I["Future production API or renderer"]
```

## Track Analysis Flow

```mermaid
flowchart TD
  A["Validate audio path"] --> B["Estimate BPM"]
  A --> C["Estimate beatgrid and downbeats"]
  A --> D["Estimate chroma key"]
  A --> E["Extract energy curve"]
  A --> F["Extract MFCC-like timbre"]
  B --> G["TrackAnalysis"]
  C --> G
  D --> G
  E --> G
  F --> G
  G --> H["JSON schema validation later"]
```

## Transition Planning Flow

```mermaid
flowchart TD
  A["Track A analysis"] --> C["Composite compatibility score"]
  B["Track B analysis"] --> C
  C --> D["Choose phrase-aligned exit"]
  C --> E["Choose phrase-aligned entry"]
  D --> F["Generate EQ and filter notes"]
  E --> F
  F --> G["Cue and loop suggestions"]
  G --> H["Beginner-readable remix recipe"]
```

## Future Integration Flow

```mermaid
flowchart LR
  A["Feature lab module"] --> B["Review schema contract"]
  B --> C["Port pure tests"]
  C --> D["Add backend adapter"]
  D --> E["Wire frontend surface"]
  E --> F["Run full repo tests"]
  F --> G["Remove duplicate temporary adapters"]
```

## Safe Boundaries

The lab code has no imports from the existing app. Future production integration should use adapters so current endpoints, stores, and UI components are not rewritten in one step.
