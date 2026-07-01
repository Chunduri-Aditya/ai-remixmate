# Feature Inventory

| Feature | DJ meaning | AI RemixMate version | Data needed | Frontend module | Backend module | MVP priority | Risks | Tests needed |
|---|---|---|---|---|---|---|---|---|
| Track analysis summary | Understand technical/musical identity | TrackAnalysis with BPM, key, energy, sections | Audio metadata and optional waveform | Analysis panel | analysis package | P0 | Bad estimates | schema and serialization |
| Beatgrid validation | Know if sync can be trusted | Ascending beatgrid with confidence | Beat times, downbeats | Waveform grid | beatgrid.py | P0 | Drift and missing beats | ascending validation |
| Downbeat alignment | Land phrases on beat one | Downbeat list and cue snap | Beatgrid phase | Timeline markers | transition planner | P0 | Wrong bar phase | entry/exit timing tests |
| BPM compatibility | Judge tempo blend risk | Normalized half/double-aware score | BPM A/B | Compatibility panel | compatibility.py | P0 | Half-time errors | same/near/double tests |
| Tempo safety warning | Avoid artifact-heavy stretch | Safe percent threshold | BPM A/B max percent | Tempo badge | tempoMath | P0 | False unsafe warnings | threshold tests |
| Camelot key scoring | Estimate harmonic blend | Same, relative, adjacent, boost rules | Camelot key | Key badge | keyCompatibility | P0 | Unknown key | rule tests |
| Energy curve scoring | Compare dynamic shape | Resampled energy distance | Energy arrays | Energy arc | energyCompatibility | P0 | Different array lengths | range tests |
| Timbre scoring | Compare sonic color | Cosine similarity over MFCC-like vectors | MFCC vectors | Breakdown row | timbreCompatibility | P1 | Sparse vectors | cosine tests |
| Vocal clash risk | Avoid vocal overlap | Overlap risk from vocal activity | Vocal activity arrays | Warning chips | vocalClashRisk | P1 | No stem data | risk tests |
| Composite compatibility | Explain overall match | Weighted score with warnings | BPM/key/energy/timbre/vocal | Score ring | compatibilityScore | P0 | Opaque score | range and breakdown |
| Transition entry cue | Choose incoming start | Phrase/downbeat based time | Track B beatgrid | Transition plan panel | transition_plan.py | P0 | No downbeats | valid time tests |
| Transition exit cue | Choose outgoing fade start | Phrase/downbeat based time | Track A beatgrid | Transition plan panel | transition_plan.py | P0 | Short track | valid time tests |
| Transition length bars | Control overlap duration | Default 16 bars, bounded | BPM and bars | Plan controls | transition planner | P0 | Too short/long | length tests |
| EQ automation notes | Prevent frequency clash | Bass swap and EQ guidance | Compatibility and stems | Automation list | planner output | P0 | Generic advice | presence tests |
| Filter automation notes | Mask entrances/exits | HP/LP sweep notes | Timing plan | Automation list | planner output | P1 | Over-filtering | presence tests |
| Cue suggestions | Make plan actionable | Mix-in, bass swap, exit cues | Beatgrid and sections | Cue panel | planner output | P0 | Off-grid cue | non-negative tests |
| Loop suggestions | Extend mixable regions | Beat loop proposal | Beatgrid | Cue loop panel | loopMath | P1 | End overflow | loop tests |
| Beginner recipe | Readable instructions | Ordered steps with why | Transition plan | Recipe panel | remix_recipe.py | P0 | Too vague | ordered steps tests |
| Stem manifest | Track stem availability | File manifest and quality fields | Stems directory | Stem mixer | stem_manifest.py | P0 | Missing stems | manifest tests |
| Stem mixer prototype | Control stem levels | Per-stem gain/mute/solo props | Stem manifest | StemMixerPrototype | future renderer | P1 | Solo conflicts | render smoke |
| Waveform prototype | Show structure visually | Peaks plus cues and playhead | Waveform summary | WaveformDeckPrototype | waveform_summary.py | P1 | Huge arrays | render smoke |
| Deck state model | Represent transport | DeckState schema and TS type | Track and tempo state | MixDeckPanel | frontend store | P0 | State drift | schema tests |
| Mixer state model | Represent mixer controls | Crossfader, channels, master | Deck channels | MixerControlsPrototype | renderer input | P0 | Out of range | schema tests |
| Crossfader math | Blend decks | Linear and equal-power gains | Position -1..1 | MixerControlsPrototype | renderer | P0 | Center law error | unit tests |
| Gain conversion | Convert dB and linear | Safe dB helpers | Gain values | Mixer controls | renderer | P0 | Zero linear | unit tests |
| EQ validation | Normalize bands | Clamp -24..+6 dB | EQ state | Mixer controls | renderer | P0 | Overboost | unit tests |
| Filter validation | Normalize cutoff/Q | Validated filter state | Filter params | Mixer controls | renderer | P1 | Bad cutoff | unit tests |
| Beat sync helper | Find beat relation | Nearest/next/previous/phase | Beat times | Sync status | beatSync | P0 | Empty grids | unit tests |
| Quantize helper | Snap actions | Nearest/next/bar functions | Beatgrid | Cue loop controls | quantize | P0 | No target | unit tests |
| Loop math | Create beat loops | Length clamp and validity | Beatgrid | Loop controls | loopMath | P0 | Fractional length | unit tests |
| Phrase math | Plan at phrases | Boundary helpers | Beat index | Timeline | phraseMath | P0 | Non-4/4 | unit tests |
| Automation interpolation | Evaluate lanes | Linear point interpolation | Automation points | Automation editor | automation | P1 | Unsorted points | unit tests |
| Automix ordering | Build queue | Greedy compatibility order | Track list | AutomixQueuePrototype | automix.py | P0 | Local optimum | all tracks once |
| Track ranking | Recommend next track | Sorted compatibility results | Source/candidates | Signal search | track_match.py | P0 | Sparse data | sort tests |
| Export manifest | Reproducible output | Sources, recipe, files | Recipe and render output | Mix vault | json_export.py | P1 | Version drift | schema tests |
| Analysis IO | Save JSON safely | Manifest writer | Track analysis dict | Operations | track_manifest.py | P1 | Invalid JSON | round trip tests |
| BPM estimator wrapper | Optional real analysis | librosa-loaded tempo | Audio file | Operations | bpm.py | P2 | Missing librosa | ImportError path |
| Beatgrid estimator wrapper | Optional grid analysis | librosa beats to seconds | Audio file | Waveform | beatgrid.py | P2 | No beats | ImportError path |
| Chroma key wrapper | Optional key analysis | Chroma mean key estimate | Audio file | Compatibility | chroma_key.py | P2 | Ambiguous key | ImportError path |
| MFCC wrapper | Optional timbre features | MFCC mean/std vector | Audio file | Breakdown | mfcc_features.py | P2 | Heavy dep | ImportError path |
| Energy wrapper | Optional energy analysis | RMS normalized curve | Audio file | Energy arc | energy_curve.py | P2 | Silence | ImportError path |
| Sections wrapper | Stub-ready sections | Energy segmented sections | Energy curve | Timeline | sections.py | P2 | Bad segmentation | shape tests |
| Waveform summary | Small waveform payload | Peaks/RMS buckets | Audio file | Waveform | waveform_summary.py | P2 | Non-WAV | ImportError path |
| React deck prototype | Visual state demo | Standalone typed component | Deck props | MixDeckPanel | future integration | P1 | No CSS tokens | manual smoke |
| Compatibility prototype | Explain scores | Score rows and warnings | CompatibilityScore | CompatibilityPanel | future integration | P1 | Overcrowding | manual smoke |
| Transition panel | Show plan | Cue, automation, warnings | TransitionPlan | TransitionPlanPanel | future integration | P1 | Plan missing | manual smoke |
| Recipe panel | Show recipe steps | Beginner-readable steps | RemixRecipe | RecipePanel | future integration | P1 | Too verbose | manual smoke |
| Codex prompts | Parallel future work | Scoped module prompts | Repo context | N/A | N/A | P1 | Agents touching root | prompt review |
| No-touch verification | Prove isolation | Git status comparison | Git output | Reports | N/A | P0 | Dirty baseline | report check |
| Blocker reporting | Keep progress safe | Blocker markdown | Errors | Reports | N/A | P0 | Silent failure | report check |
| Integration plan | Move safely later | Read-only path map | Repo summary | Docs | N/A | P0 | Big bang merge | doc review |
| Testing plan | Define quality bar | TS and Python commands | Test suite | Docs | CI later | P0 | Missing deps | doc review |
