# RemixMate provision runbook (Phase 7)

Code on `hosted-app` is deploy-ready after Phases 0–8. This runbook is the
**operator** path that unlocks the resume line *"deployed at [URL]"*. Until every
live-verify box is green against real output, the honest claim stays
**Dockerized**.

## 1. Accounts and hardware

1. Buy a Hetzner CX22 (~€4.5/mo) + ~60 GB volume (~€3/mo).
2. Create a Supabase project (free tier is enough for demo traffic).
3. Create a Cloudflare Pages project for the React frontend (free).
4. Point DNS: `api.<domain>` → VPS A record; `<domain>` → Pages.

## 2. VPS bootstrap

```bash
# as root
apt update && apt install -y ffmpeg libsndfile1 python3.12-venv caddy postgresql
useradd -r -m -d /opt/remixmate remixmate
# mount catalog volume read-only at /mnt/catalog
git clone <repo> /opt/remixmate && cd /opt/remixmate && git checkout hosted-app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
# do NOT install [generative] / [wordplay] extras on the box
```

Copy [`remixmate.env.example`](remixmate.env.example) → `/etc/remixmate.env`, fill
secrets, `chmod 600`. Install unit + Caddy:

```bash
cp app/deploy/remixmate.service /etc/systemd/system/
# edit app/deploy/Caddyfile: replace YOUR_DOMAIN, then:
cp app/deploy/Caddyfile /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl enable --now remixmate caddy
```

Optional: `systemd-analyze verify /etc/systemd/system/remixmate.service`

## 3. Frontend (Cloudflare Pages)

Build with:

- `VITE_API_BASE=https://api.<domain>`
- `VITE_EVENTS_URL=https://api.<domain>/events/stream`
- `VITE_SUPABASE_URL=https://<ref>.supabase.co`
- `VITE_SUPABASE_ANON_KEY=…`

Pair with API `REMIXMATE_AUTH_ENABLED=true` + matching JWKS URL.

## 4. Catalog

Curate ~150 CC tracks (ccMixter + Internet Archive). Run Demucs at ingest
(~35 GB). `rsync` the built volume to `/mnt/catalog` (read-only mount). Catalog
is **not** part of the git deploy.

## 5. Backups

Nightly: `pg_dump $DATABASE_URL | gzip` → object storage (Backblaze/R2).

## 6. Live verify (must all pass)

- [ ] Sign up, log in, render a mix end to end; download includes `CREDITS.txt`
- [ ] `systemctl restart remixmate` drains in-flight jobs; `/health/ready` returns 200
- [ ] `pg_dump` dumps and restores clean into a scratch DB
- [ ] **Cross-tenant:** a second account cannot list or fetch the first account's mixes / crates / jobs
- [ ] Median mix render time on the CX22 is acceptable to show a recruiter

Only after every box is ticked: update the resume from "Dockerized" to
"deployed at [URL]".

## 7. CI

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs ruff, the
pytest safe subset, and frontend `tsc` / vitest / vite build on PRs. Leave
`pages.yml` and `codeql.yml` disabled unless you intentionally re-enable them.
