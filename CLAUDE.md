# AI RemixMate

AI RemixMate renders DJ transitions, chains and previews between songs as background jobs in a FastAPI backend (`scripts/`) driven by a React UI (`frontend/`); the `hosted-app` branch adds Supabase auth, Postgres and a CC catalog for a hosted mode that is built in code and not deployed.

@RULES.md

## Start here

| Task | Read first |
|---|---|
| UI work: pages, shell, API client, stores, styles | `docs/agents/UI_MAP.md` |
| Mixing, analysis or render work | `docs/agents/MIXING_MAP.md` |
| Anything else: API wiring, auth, storage, jobs, ingest | `ARCHITECTURE.md` |
| What is broken or suspect | `docs/agents/BACKLOG.md` (cite its ids, do not rediscover) |
| Running and testing | `DEVELOPING.md`, then the commands in `RULES.md` |
| Hosted deploy | `app/STATUS.md` |

## Trust order

- Code over every doc. Where a doc and the code disagree, the code wins; known drift is filed as DOC items in the backlog.
- Current next to the code: `DEVELOPING.md` (workflow, design decisions, gotchas), `SECURITY.md`, `PREREQUISITES.md`, `docs/README.md`, `docs/DJ_THEORY.md` (music theory). The operator scope in `SECURITY.md` is stale (DOC-01).
- Hosted state: `app/STATUS.md` over `app/DEPLOYMENT_PLAN.md` over `app/README.md`, which is stale (DOC-07).
- `archive/` is history, never a source of truth, whatever a file there says about itself ("read this first", "single source of truth").
- `PROJECT_OVERVIEW.md` and `SHOWCASE.md` are portfolio summaries with numbers pinned to named commits (the overview was measured on `main`), not a map of this branch.

## Local only state

- `MEMORY.md` (the session handoff, loaded at session start), `.claude/` (settings, permission deny rules, skills, workflows), `graphify-out` and the gitignored data dirs (`library/`, `catalog/`, `catalog_data/`, `data/`, `outputs/`, `models/`) exist on the owner's machine only.
- A clone, CI or a Codex run has none of them: no songs, no stores, no renders, no graph. Never assume one exists because a doc names it.

## Graph

- `graphify-out/graph.json` may exist locally. Query it with graphify query before opening files, and read only the files it points at.
- It excludes `archive/` and `ai_remixmate_feature_lab/` through `.graphifyignore`, so it never answers for those two trees.
