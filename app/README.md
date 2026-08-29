# app/ — hosted multi-tenant product layer

This directory is the home for turning ai-remixmate (a single-user local tool)
into a **low-traffic multi-user hosted product** on a ~$10-30/mo budget, per
[`DEPLOYMENT_PLAN.md`](./DEPLOYMENT_PLAN.md). Work happens on the `hosted-app`
branch.

## What lives here

- [`DEPLOYMENT_PLAN.md`](./DEPLOYMENT_PLAN.md) — the canonical phased roadmap
  (Status table, per-phase task + verify checklists). Keep this in sync with the
  work; it is the source of truth for what is done.
- `deploy/` — production infra that has no other natural home: `Caddyfile`,
  `remixmate.service` (systemd unit), `remixmate.env.example`, provisioning
  notes. Populated in Phase 7.
- Product docs / legal copy source, serverless stubs (e.g. a Modal Demucs
  function) if/when they exist.

## What does NOT live here

The plan **transforms the existing codebase in place** — it is surgery, not an
add-on. So these stay where they are:

- The catalog-ingest rework, license gate, per-user contextvar, auth dependency,
  Postgres engine — all edits to `scripts/` (`download.py`, `core/license.py`,
  `api/jobs.py`, new `api/auth.py`, new `core/db.py`, …). New modules sit next to
  their package siblings, not here, so imports stay clean.
- Frontend changes — `frontend/src/`.
- CI — `.github/workflows/ci.yml` (GitHub mandates that path).
- Test isolation — `tests/conftest.py`.

## Status

See the Status table at the top of [`DEPLOYMENT_PLAN.md`](./DEPLOYMENT_PLAN.md).
Phase 0 (test storage isolation) is done and verified.
