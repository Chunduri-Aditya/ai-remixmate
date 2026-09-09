# app/deploy/

Production infra files for the hosted multi-user stack (Phase 7).

| File | Purpose |
|---|---|
| [`bootstrap.sh`](bootstrap.sh) | idempotent VPS setup: packages, user, venv, systemd/Caddy install |
| [`backup.sh`](backup.sh) | pg_dump + gzip to local/remote storage; safe for nightly cron |
| [`Caddyfile`](Caddyfile) | reverse proxy + automatic TLS → `uvicorn` on 127.0.0.1:8000 |
| [`remixmate.service`](remixmate.service) | systemd unit: 1 worker, `EnvironmentFile=/etc/remixmate.env`, `TimeoutStopSec=600` drain |
| [`remixmate.env.example`](remixmate.env.example) | template for `/etc/remixmate.env` (chmod 600) — placeholders only |
| [`PROVISION.md`](PROVISION.md) | operator runbook + live-verify checklist |

The catalog (audio + stems) ships by a separate `rsync` to a read-only volume,
not through git. See [`PROVISION.md`](PROVISION.md) before spending on hardware.
