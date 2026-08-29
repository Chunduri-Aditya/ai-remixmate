# app/deploy/

Production infra files. **Populated in Phase 7** of
[`../DEPLOYMENT_PLAN.md`](../DEPLOYMENT_PLAN.md). Empty until then.

Planned contents:

| File | Purpose |
|---|---|
| `Caddyfile` | reverse proxy + automatic TLS in front of `uvicorn` on the VPS |
| `remixmate.service` | systemd unit: 1 `uvicorn` worker, `EnvironmentFile=/etc/remixmate.env`, `ExecStop` drain with `TimeoutStopSec=600` |
| `remixmate.env.example` | template for `/etc/remixmate.env` (chmod 600 on the box) — every `REMIXMATE_*` / Supabase / `DATABASE_URL` key with a placeholder value, no secrets committed |
| `PROVISION.md` | fresh-VPS runbook: `apt` packages, venv, catalog volume mount (read-only), first deploy |

The catalog itself (audio + pre-separated stems) ships by a separate `rsync` to
the read-only volume, not through git.
