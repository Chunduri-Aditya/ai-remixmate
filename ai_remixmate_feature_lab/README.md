# AI RemixMate Feature Lab

This folder is an isolated feature laboratory for AI RemixMate. It contains a complete, dependency-light implementation of remix-intelligence models, schemas, audio math primitives, deterministic scoring logic, backend-ready Python modules, frontend React prototypes, tests, Codex prompts, and integration notes.

The lab is intentionally separate from the production project. Existing files in the repository are read-only for this work. Nothing here imports from `scripts/`, `frontend/src/`, or any root package file. Future integration should copy or adapt reviewed pieces from this folder into the production codebase in small pull requests after tests and API contracts are agreed.

## What Is Implemented

- JSON Schemas for core remix objects such as `TrackAnalysis`, `BeatGrid`, `DeckState`, `MixerState`, `TransitionPlan`, `RemixRecipe`, stems, effects, automation, and export manifests.
- Strict TypeScript model interfaces that mirror those schemas.
- Pure TypeScript audio primitives for gain, EQ, filters, crossfaders, tempo math, beat sync, quantize, loops, phrases, and automation interpolation.
- Pure TypeScript intelligence modules for BPM, key, energy, timbre, vocal-clash, composite compatibility, transition planning, remix recipes, automix, and track ranking.
- Standalone React prototypes for deck, waveform, mixer, cue/loop, compatibility, transition-plan, recipe, stem mixer, and automix queue UI surfaces.
- Python dataclasses, analysis wrappers with optional heavy dependencies, pure Python intelligence modules, IO helpers, and pytest tests.
- Dependency-free online neural learning modules for each adaptive remix feature. These models learn only from explicit `TrainingEvent` feedback and persist JSON weights inside the lab.
- Codex prompts for future isolated sub-work.
- Reports describing implementation status, blockers, and no-touch verification.

## Isolation Rules

All files created for this feature lab live under `ai_remixmate_feature_lab/`. The lab does not modify root dependencies, lockfiles, frontend source, backend source, configs, data, or tests. The code is meant to be evaluated here first, then integrated deliberately.

## TypeScript Tests

The TypeScript package is self-contained. If dependencies are available locally, run:

```bash
cd ai_remixmate_feature_lab/typescript
npm test
```

If dependencies have not been installed, either install them inside this isolated folder or run with an already available Vitest binary. Do not run package managers from the repository root for this lab.

```bash
cd ai_remixmate_feature_lab/typescript
../../frontend/node_modules/.bin/vitest run --config vitest.config.ts
```

## Python Tests

The Python package uses only the standard library for intelligence tests. Heavy audio dependencies such as librosa are imported only inside optional analysis functions.

```bash
cd ai_remixmate_feature_lab/python
python -m pytest -p no:cacheprovider
```

## Safe Future Integration

1. Treat schemas as the first contract. Compare them to current FastAPI Pydantic schemas before adding endpoints.
2. Port pure intelligence modules before UI. They have low integration risk and deterministic tests.
3. Keep neural learning behind explicit feedback events. Do not enable background training or opaque score replacement.
4. Add backend endpoints behind feature flags or new namespaces rather than changing existing routes in place.
5. Wire React prototypes into existing pages only after matching the app design tokens and store shape.
6. Run repository-level tests only after a reviewed integration branch exists. This lab itself should not mutate root project files.
