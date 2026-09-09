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

After cloning the repo to `/opt/remixmate`, run:

```bash
sudo bash app/deploy/bootstrap.sh
```

The script is idempotent and installs packages, creates the system user, sets up
the Python environment, and copies systemd/Caddy files. Follow on-screen prompts
to fill `/etc/remixmate.env` (from `remixmate.env.example`) and configure the
`Caddyfile` with your domain.

**Manual fallback** (if you prefer step-by-step):

```bash
# as root
apt update && apt install -y ffmpeg libsndfile1 python3.12-venv caddy postgresql
useradd -r -m -d /opt/remixmate remixmate
# mount catalog volume read-only at /mnt/catalog
cd /opt/remixmate && git checkout hosted-app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
# do NOT install [generative] / [wordplay] extras on the box

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

Run [`app/deploy/backup.sh`](backup.sh) nightly via cron. Set `RCLONE_REMOTE` to
upload to object storage (Backblaze/R2). Example crontab:

```cron
0 2 * * * /opt/remixmate/app/deploy/backup.sh >> /var/log/remixmate-backup.log 2>&1
```

## 6. Live verify (must all pass)

Prefer env vars so JWTs never land in shell history:

```bash
export LIVE_VERIFY_TOKEN_A='…'   # access token for user A
export LIVE_VERIFY_TOKEN_B='…'   # access token for a different user B
python -m scripts.system.live_verify \
  --base-url https://api.<domain> \
  --song-a "<catalog_song_name>" \
  --song-b "<catalog_song_name>"
```

All harness checks must pass (exit 0), including cross-tenant crate isolation.
Manual checklist:

- [ ] Sign up, log in, render a mix end to end; download includes `CREDITS.txt`
- [ ] `systemctl restart remixmate` drains in-flight jobs; `/health/ready` returns 200
- [ ] `pg_dump` dumps and restores clean into a scratch DB
- [ ] **Cross-tenant:** harness exit 0 (and second account cannot list first account's mixes)
- [ ] Median mix render time on the CX22 is acceptable to show a recruiter

Only after every box is ticked: update the resume from "Dockerized" to
"deployed at [URL]".

## 7. CI

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs ruff, the
pytest safe subset, and frontend `tsc` / vitest / vite build on PRs. Leave
`pages.yml` and `codeql.yml` disabled unless you intentionally re-enable them.
