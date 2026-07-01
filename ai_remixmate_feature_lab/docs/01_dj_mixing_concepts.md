# DJ Mixing Concepts

Each concept is modeled for an AI-assisted remix engineering workflow. The goal is to preserve generic DJ meaning while turning it into typed, testable data and deterministic planning logic.

## Deck

- Definition: A logical player that holds one track and its transport state.
- Why DJs use it: DJs use decks to stage, cue, and mix multiple tracks.
- How AI RemixMate should model it: Represent as `DeckState` with loaded track, play state, tempo, pitch, cue, loop, and sync flags.
- Frontend implications: Deck panels need clear loaded-track metadata, transport state, cue/loop state, and sync warnings.
- Backend/ML implications: Backend can persist deck-independent analysis; live deck state belongs in frontend/session state.
- Edge cases: No loaded track, bad BPM, missing beatgrid, or deck running without quantize.
- MVP version: Two typed decks, A and B, with selected tracks and BPM/key display.
- Advanced version: Multi-deck state with per-deck stem routing and automation.

## Mixer

- Definition: The control surface that combines deck channels into a master output.
- Why DJs use it: DJs use mixers to balance levels, EQ, filters, and crossfader position.
- How AI RemixMate should model it: Represent as `MixerState` with channels, crossfader, master gain, cue mix, and limiter flags.
- Frontend implications: Mixer controls need stable sliders/meters and visible gain/EQ ranges.
- Backend/ML implications: Backend renderer can consume mixer snapshots for offline exports.
- Edge cases: Clipping, mismatched channel IDs, crossfader outside range.
- MVP version: Two channels plus crossfader and master volume.
- Advanced version: Stem-aware mixer with automation lanes and limiter telemetry.

## Channel

- Definition: One signal path through the mixer for a deck or stem group.
- Why DJs use it: Channels let DJs adjust each source independently.
- How AI RemixMate should model it: Represent each channel with gain, EQ, filter, mute, solo, and meter values.
- Frontend implications: Channel strips should be dense and scan-friendly.
- Backend/ML implications: Renderer can map channel values to gain and filter DSP.
- Edge cases: Muted but still metered signals, solo conflicts, stale meters.
- MVP version: Deck A and B channels.
- Advanced version: Per-stem channels for drums, bass, vocals, and other.

## Gain

- Definition: Input trim before channel volume.
- Why DJs use it: DJs match loudness before mixing so faders behave predictably.
- How AI RemixMate should model it: Store gain in dB and convert with tested math.
- Frontend implications: Use dB sliders with center detent and clipping hints.
- Backend/ML implications: Analysis can suggest gain from loudness estimates.
- Edge cases: Extreme boost can clip; silence maps to safe floor.
- MVP version: Clamp gain and expose dB conversion.
- Advanced version: Auto-gain from LUFS and true-peak estimates.

## Volume fader

- Definition: Post-EQ level for one channel.
- Why DJs use it: DJs fade tracks in and out without changing trim.
- How AI RemixMate should model it: Represent as normalized 0-1 channel volume.
- Frontend implications: Vertical or horizontal fader with stable size.
- Backend/ML implications: Renderer maps fader automation to sample gain curves.
- Edge cases: Fader jumps, automation discontinuities.
- MVP version: Manual normalized fader.
- Advanced version: Editable automation curves.

## Master volume

- Definition: Final output level after all channels.
- Why DJs use it: DJs control room or recording output.
- How AI RemixMate should model it: Store normalized master gain plus limiter state.
- Frontend implications: Display master meter and limiter warning.
- Backend/ML implications: Mastering can normalize export target.
- Edge cases: Clipping when channels sum hot.
- MVP version: Master scalar.
- Advanced version: LUFS-aware output stage.

## Crossfader

- Definition: A single control that blends left and right sides.
- Why DJs use it: DJs perform quick or smooth transitions with one motion.
- How AI RemixMate should model it: Provide linear and equal-power gain laws from -1 to 1.
- Frontend implications: Crossfader UI should show left/right gain relation.
- Backend/ML implications: Renderer can apply equal-power fade over transition windows.
- Edge cases: Wrong center gain, abrupt jumps, inverse mapping.
- MVP version: Pure crossfade math.
- Advanced version: Curve selection and stem-specific crossfades.

## EQ high/mid/low

- Definition: Tone controls for frequency bands.
- Why DJs use it: DJs avoid clashes and shape blend energy.
- How AI RemixMate should model it: Clamp each band from -24 dB to +6 dB.
- Frontend implications: Three compact knobs/sliders per channel.
- Backend/ML implications: Renderer maps EQ state to filters or stem gains.
- Edge cases: Bass clash, overboost, unclear neutral state.
- MVP version: Validated EQ state.
- Advanced version: Dynamic EQ automation and band metering.

## Low-pass filter

- Definition: Filter that keeps low frequencies and removes highs above cutoff.
- Why DJs use it: Used for sweeps, tension, and soft transitions.
- How AI RemixMate should model it: Represent cutoff and Q, no DSP in primitives.
- Frontend implications: Filter knob with active state and cutoff display.
- Backend/ML implications: Backend DSP can later consume validated state.
- Edge cases: Cutoff outside audible range, resonance too high.
- MVP version: Validate cutoff/Q.
- Advanced version: Automated filter sweeps.

## High-pass filter

- Definition: Filter that keeps highs and removes lows below cutoff.
- Why DJs use it: Used to remove bass before bringing in another track.
- How AI RemixMate should model it: Represent cutoff and Q separately from EQ.
- Frontend implications: Show HPF state near channel EQ.
- Backend/ML implications: Renderer can use HPF to prevent bass clash.
- Edge cases: Thin sound if cutoff stays too high.
- MVP version: Validate cutoff/Q.
- Advanced version: Phrase-timed HPF automation.

## BPM

- Definition: Beats per minute, the tempo estimate.
- Why DJs use it: DJs use BPM to match track speeds.
- How AI RemixMate should model it: Positive numeric field with confidence and normalization helpers.
- Frontend implications: Show BPM with decimals and warnings for uncertain values.
- Backend/ML implications: Analysis estimates BPM and resolves half/double ambiguity.
- Edge cases: Half-time/double-time errors and variable tempo.
- MVP version: Positive BPM and compatibility score.
- Advanced version: Tempo map with local BPM segments.

## Tempo

- Definition: Playback speed relative to original.
- Why DJs use it: DJs adjust tempo to align beats.
- How AI RemixMate should model it: Represent as percent change and ratio.
- Frontend implications: Show percent change with safe/unsafe coloring.
- Backend/ML implications: Renderer uses stretch ratio for time-scaling.
- Edge cases: Large tempo shifts cause artifacts.
- MVP version: Safe stretch threshold.
- Advanced version: Elastique/phase-vocoder strategy selection.

## Beatgrid

- Definition: Ordered beat timestamps for a track.
- Why DJs use it: DJs rely on grids for sync, loops, and quantized cues.
- How AI RemixMate should model it: Store ascending beat times, downbeats, BPM, beats per bar, confidence.
- Frontend implications: Waveform should render grid lines and downbeats distinctly.
- Backend/ML implications: Analysis produces beat timestamps and confidence.
- Edge cases: Missing first downbeat, drift, non-ascending times.
- MVP version: Static beat array.
- Advanced version: Dynamic beatgrid with drift correction.

## Downbeat

- Definition: First beat of a bar.
- Why DJs use it: DJs align downbeats so phrases land musically.
- How AI RemixMate should model it: Store downbeat timestamps as part of BeatGrid.
- Frontend implications: Render stronger grid markers at downbeats.
- Backend/ML implications: Analysis infers bar phase and phrase starts.
- Edge cases: Wrong phase makes transitions feel late.
- MVP version: Every fourth beat fallback.
- Advanced version: Model-predicted downbeats.

## Measure/bar

- Definition: A group of beats, usually four in dance music.
- Why DJs use it: DJs count bars to plan transitions.
- How AI RemixMate should model it: Use beatsPerBar and bar index math.
- Frontend implications: Show bar counts in transition plans.
- Backend/ML implications: Backend converts bars to seconds from beatgrid.
- Edge cases: 3/4, 6/8, or irregular structures.
- MVP version: Default four beats per bar.
- Advanced version: Time-signature-aware sections.

## Phrase

- Definition: A larger musical unit, often 8 or 16 bars.
- Why DJs use it: DJs start transitions at phrase boundaries.
- How AI RemixMate should model it: Estimate phrase indexes and next boundaries from beat indexes.
- Frontend implications: Timeline should show phrase blocks.
- Backend/ML implications: Section analysis can refine phrase boundaries.
- Edge cases: Breakdowns, pickups, and skipped phrases.
- MVP version: Math-based phrase boundaries.
- Advanced version: Novelty-detection phrase segmentation.

## Sync

- Definition: Keeping decks rhythmically aligned.
- Why DJs use it: DJs use sync to reduce manual beatmatching load.
- How AI RemixMate should model it: Store syncEnabled plus target deck and beat phase.
- Frontend implications: Show when sync is active and what it follows.
- Backend/ML implications: Backend plans can compute offsets; live sync belongs in frontend/audio engine.
- Edge cases: Bad grids make sync harmful.
- MVP version: Beat phase helper.
- Advanced version: Continuous tempo and phase correction.

## Tempo sync

- Definition: Matching BPM only.
- Why DJs use it: Useful when grids are uncertain but tempos are close.
- How AI RemixMate should model it: Use BPM ratio and tempo percent functions.
- Frontend implications: Show tempo delta separately from beat phase.
- Backend/ML implications: Renderer uses time-stretch ratio.
- Edge cases: Half/double ambiguity.
- MVP version: Normalize BPM relation.
- Advanced version: Tempo-map sync.

## Beat sync

- Definition: Matching beat phase and tempo.
- Why DJs use it: Creates tight transitions when grids are reliable.
- How AI RemixMate should model it: Use nearest/next/previous beat and phase offset helpers.
- Frontend implications: Show phase offset and quantize target.
- Backend/ML implications: Renderer aligns entry samples to downbeats.
- Edge cases: Wrong grid creates flamming.
- MVP version: Phase offset in seconds.
- Advanced version: Sample-level correction.

## Tempo bend/nudge

- Definition: Temporary speed adjustment to align beats.
- Why DJs use it: DJs fix small drift manually.
- How AI RemixMate should model it: Represent future intent as deck nudge amount, not DSP yet.
- Frontend implications: Small nudge buttons or wheel gestures.
- Backend/ML implications: Backend likely does not persist nudge except automation.
- Edge cases: Overcorrection, accumulating drift.
- MVP version: Documented state field.
- Advanced version: Real-time transport controller.

## Quantize

- Definition: Snapping actions to beats or bars.
- Why DJs use it: Prevents cues and loops from firing off-grid.
- How AI RemixMate should model it: Quantize helper returns nearest, next beat, or bar time.
- Frontend implications: Cue/loop controls should show quantize mode.
- Backend/ML implications: Backend can snap generated cues to beatgrid.
- Edge cases: Sparse beatgrid, action before first beat.
- MVP version: Beat/bar quantize helpers.
- Advanced version: User-selectable quantize resolution.

## Cue point

- Definition: Saved timestamp for jumping or planning.
- Why DJs use it: DJs mark mix-in, vocal, drop, and exit points.
- How AI RemixMate should model it: CuePoint includes timestamp, label, type, color, confidence.
- Frontend implications: Cue panel lists type and time.
- Backend/ML implications: Analysis can suggest cues from sections.
- Edge cases: Duplicate cues, off-grid cues.
- MVP version: Typed cue list.
- Advanced version: AI-labeled cue generation.

## Hot cue

- Definition: Performance cue triggered instantly.
- Why DJs use it: DJs jump to important moments live.
- How AI RemixMate should model it: CuePoint type `hot`.
- Frontend implications: Buttons with labels and colors.
- Backend/ML implications: Backend stores metadata only.
- Edge cases: Jumping mid-vocal or off-grid.
- MVP version: Static hot cues.
- Advanced version: Quantized hot cue launch.

## Loop

- Definition: Repeated region between start and end.
- Why DJs use it: DJs extend intros/outros and build transitions.
- How AI RemixMate should model it: LoopState includes start, end, length in beats, active flag.
- Frontend implications: Loop controls need halve/double and valid state.
- Backend/ML implications: Backend can suggest loop windows.
- Edge cases: End before start, too short, too long.
- MVP version: Loop validation and beat loop creation.
- Advanced version: Slip-aware loop rolls.

## Loop halve/double

- Definition: Changing loop length by powers of two.
- Why DJs use it: DJs tighten or extend loops during transitions.
- How AI RemixMate should model it: Clamp loop length from 1/16 to 32 beats.
- Frontend implications: Use dedicated halve/double buttons.
- Backend/ML implications: Renderer can create repeated regions later.
- Edge cases: Boundary overflows and fractional beats.
- MVP version: Pure length functions.
- Advanced version: Quantized loop resizing on beat.

## Slip mode

- Definition: Playback continues silently while effects or loops happen.
- Why DJs use it: DJs exit tricks without losing song position.
- How AI RemixMate should model it: Represent as future deck flag.
- Frontend implications: Toggle only when supported.
- Backend/ML implications: Requires transport engine support.
- Edge cases: Confusing if visual playhead does not show real position.
- MVP version: Document field only.
- Advanced version: Full shadow transport implementation.

## Slicer mode

- Definition: Chops a phrase into repeatable slices.
- Why DJs use it: DJs remix rhythmic fragments live.
- How AI RemixMate should model it: Represent as future mode using beatgrid slices.
- Frontend implications: Grid of slice pads.
- Backend/ML implications: Needs reliable beatgrid and sampler/playback engine.
- Edge cases: Bad grids create clicks or off-time slices.
- MVP version: Planning concept only.
- Advanced version: Beat-sliced performance surface.

## Key

- Definition: Musical tonal center and mode.
- Why DJs use it: DJs avoid harmonic clashes.
- How AI RemixMate should model it: Store key, mode, Camelot, confidence.
- Frontend implications: Show key and Camelot badge.
- Backend/ML implications: Analysis estimates chroma key.
- Edge cases: Ambiguous key, modal interchange.
- MVP version: String key plus Camelot.
- Advanced version: Continuous harmonic embeddings.

## Key lock

- Definition: Preserving pitch while tempo changes.
- Why DJs use it: DJs change tempo without changing musical key.
- How AI RemixMate should model it: Represent as deck setting for future playback.
- Frontend implications: Toggle near tempo control.
- Backend/ML implications: Renderer should time-stretch independently from pitch shift.
- Edge cases: Artifacts at extreme tempo shifts.
- MVP version: State field only.
- Advanced version: High-quality pitch/time engine.

## Key shifting

- Definition: Changing pitch by semitones.
- Why DJs use it: DJs align keys when tempo/key relation is close.
- How AI RemixMate should model it: Store semitone shift and warning.
- Frontend implications: Show suggested shift with risk.
- Backend/ML implications: Renderer can apply pitch shift later.
- Edge cases: Vocal artifacts, formant change.
- MVP version: Suggested shift field.
- Advanced version: Stem-specific pitch shift.

## Harmonic mixing

- Definition: Choosing tracks with compatible keys.
- Why DJs use it: DJs use it to make blends sound musical.
- How AI RemixMate should model it: Camelot rules score same, relative, adjacent, and energy moves.
- Frontend implications: Compatibility panel should explain relation.
- Backend/ML implications: Backend can combine key with chroma confidence.
- Edge cases: Unknown key, false detections.
- MVP version: Simplified Camelot.
- Advanced version: Tonal interval space scoring.

## Waveform

- Definition: Visual summary of audio amplitude and structure.
- Why DJs use it: DJs see drops, breaks, and phrasing.
- How AI RemixMate should model it: Waveform summary includes peaks, RMS, duration, sample rate.
- Frontend implications: Deck waveform should show playhead, cues, loops, beats.
- Backend/ML implications: Backend can precompute downsampled peaks.
- Edge cases: Huge files, silent sections.
- MVP version: Prototype bars from provided values.
- Advanced version: Multi-band or stem waveforms.

## Stem

- Definition: Separated source component like drums, bass, vocals, other.
- Why DJs use it: DJs remix parts independently.
- How AI RemixMate should model it: StemManifest tracks paths, quality, duration, and availability.
- Frontend implications: Stem mixer needs per-stem mute/solo/gain.
- Backend/ML implications: Backend uses Demucs outputs and stem-aware renderer.
- Edge cases: Missing stems, mismatched duration, bleed.
- MVP version: Manifest and prototype mixer.
- Advanced version: Stem-level automation and loudness.

## Sampler

- Definition: Triggerable short audio clips.
- Why DJs use it: DJs add one-shots or loops.
- How AI RemixMate should model it: Represent as future clip metadata outside current MVP.
- Frontend implications: Pad grid with sample labels.
- Backend/ML implications: Needs file management and playback.
- Edge cases: Licensing and timing.
- MVP version: Document integration point.
- Advanced version: Quantized sample launcher.

## Effects

- Definition: Audio transformations like echo, reverb, filter, gate.
- Why DJs use it: DJs add motion and mask transitions.
- How AI RemixMate should model it: EffectState includes type, enabled, wet, and parameters.
- Frontend implications: Effects panel should show enabled state and wet amount.
- Backend/ML implications: Renderer can map validated parameters to DSP.
- Edge cases: Overuse, clipping, unsupported params.
- MVP version: Typed state only.
- Advanced version: Automated effect chains.

## Automix

- Definition: Automatic ordering and transition planning.
- Why DJs use it: DJs use it for unattended or assisted sequencing.
- How AI RemixMate should model it: Greedy planner ranks next tracks by compatibility.
- Frontend implications: Queue should show confidence and warnings.
- Backend/ML implications: Backend can create chain remix jobs later.
- Edge cases: Local optimum, repetitive energy.
- MVP version: Deterministic greedy order.
- Advanced version: Beam search with energy arc constraints.

## Track match

- Definition: Finding compatible next tracks.
- Why DJs use it: DJs use recommendations while building sets.
- How AI RemixMate should model it: Rank candidates by composite compatibility.
- Frontend implications: Show score breakdown and reasons.
- Backend/ML implications: Backend can use index plus compatibility rerank.
- Edge cases: Sparse metadata, overfitting to BPM.
- MVP version: Sorted candidate list.
- Advanced version: Hybrid vector and rule-based search.

## Recording/export

- Definition: Saving the output or plan.
- Why DJs use it: DJs need shareable mixes and reproducible recipes.
- How AI RemixMate should model it: ExportManifest captures source tracks, recipe, rendered file, and metadata.
- Frontend implications: Export panel should show plan and file status.
- Backend/ML implications: Backend renderer writes audio and manifest.
- Edge cases: Missing source files, version drift.
- MVP version: JSON manifest.
- Advanced version: Audio render plus audit trail.


## Manual-Informed Coverage

A local djay Pro Mac manual was used as a generic DJ-feature checklist. The concepts above intentionally cover the same broad professional DJ categories - decks, mixer, master and headphone cueing, beatgrid edits, BPM half/double correction, key lock and semitone shifting, tempo sync versus beat sync, slicer/slip modes, library preview and queue, Automix, Track Match, effects, loops, cue points, sampler pads, and recording/export - while keeping AI RemixMate's implementation focused on stem-aware remix planning and compatibility intelligence.
