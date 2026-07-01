# Product Direction

AI RemixMate should be an AI-assisted remix engineering lab, not a clone of a DJ application. A traditional DJ app centers on real-time playback controls. This lab centers on analysis, planning, explanation, and exportable remix intent that can later drive playback or rendering.

## Product Identity

The core identity is stem-aware remix intelligence. The system helps a user understand how two or more tracks can fit together by looking at tempo, harmonic relation, beatgrid quality, energy movement, timbre, vocal density, stem availability, and phrase timing. The result should be a plan that a beginner can read and an engineer can execute.

## What It Helps Users Do

- Analyze tracks into beatgrids, downbeats, phrases, keys, energy curves, waveform summaries, and stem manifests.
- Compare track compatibility with explanations rather than a single opaque score.
- Beatmatch and key-match with clear warnings about unsafe tempo shifts or harmonic risk.
- Plan transitions at phrase boundaries using EQ, filter, stem, cue, loop, and automation suggestions.
- Create remix recipes that describe what to do, when to do it, and why.
- Rank candidate tracks for a set or automix queue.
- Export structured plans for future backend jobs, UI timelines, or renderer automation.

## What It Should Not Become

The lab should not copy the surface area of existing DJ software. It does not need turntable skins, branded control layouts, proprietary workflow terms, or hardware-mapped assumptions. Generic deck, mixer, waveform, cue, loop, stem, and transition concepts are enough.

## Experience Principles

- Explain the musical reason behind each recommendation.
- Show risk separately from opportunity.
- Keep generated plans editable.
- Make beginner language available without removing technical detail.
- Keep analysis and scoring deterministic before introducing learned models.
- Let stem awareness be a first-class concept, not an afterthought.

## MVP Definition

The MVP is a local, testable package that accepts track-like metadata, computes compatibility, creates transition plans, creates remix recipes, ranks candidates, and demonstrates UI surfaces with typed props. It does not need live audio playback or root app integration.

## Advanced Direction

Advanced versions can add learned beatgrid confidence, neural key estimation, stem-level loudness and vocal-activity maps, phrase-aware waveform rendering, editable automation lanes, and renderer execution. Those should be integrated only after schema and test coverage are stable.


## Reference Note

The local `djay-pro-mac-manual-loqual.pdf` was consulted for generic DJ workflow coverage: deck/mixer/library layout, beatgrid correction, tempo/key/sync distinctions, cue and loop workflows, effects modes, sampler pads, Automix queues, Track Match-style recommendations, and recording/export expectations. This lab uses those concepts only as general DJ references and does not copy product-specific UI or text.
