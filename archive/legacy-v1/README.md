# legacy-v1 — the 2025 Streamlit/monolith generation of ai-remixmate

This is the first generation of the project: a Streamlit `app.py` monolith with a
`src/remixmate/` package (timeline planner/renderer) and flat processing scripts.
It was superseded by the FastAPI + React project that now owns this repository
and the GitHub remote. Archived 2026-07-07 from a stale clone that had been
moved into the main project folder (originally `~/Downloads/ai-remixmate`).

## Contents

- `source/` — full working-tree snapshot at archival time, minus `.venv`,
  caches, and empty runtime placeholder dirs. Includes ~1,500 lines of
  uncommitted `src/remixmate/` rework and `PROJECT_WORKING_ARCHITECTURE.md`
  that existed nowhere else.
- `legacy-v1-full-history.bundle` — the complete git history (3 commits; the
  final one, `eca38db`, is the snapshot commit capturing all previously
  uncommitted work).

## Restoring

```bash
git clone legacy-v1-full-history.bundle legacy-v1-restored
```

Do not point a clone of this bundle at the live GitHub remote — its history is
unrelated to the current project's, and a push would clobber it.
