# AI RemixMate Feature Lab Agent Instructions

Scope is strict. Work only inside `ai_remixmate_feature_lab/` unless the user explicitly gives a new instruction that changes the allowed folder.

Existing repository files may be read for context, but must not be modified, renamed, moved, reformatted, deleted, auto-fixed, staged, committed, or branched from this folder's work.

When adding logic in this lab:

- Add or update tests in the same lab package.
- Prefer deterministic pure functions for audio math and remix intelligence.
- Keep browser audio, file IO, network calls, and heavy ML dependencies out of base tests.
- Import optional audio dependencies such as librosa inside the function that needs them and raise a clear `ImportError` when unavailable.
- Keep schemas, TypeScript models, and Python dataclasses aligned.
- Do not run package managers from the repository root.
- Do not touch root lockfiles.
- Before final response, run Git verification from the repository root and report whether any changes outside this folder appeared.

If blocked, document the blocker in `reports/blockers.md` and continue with every independent task that can be completed safely.
