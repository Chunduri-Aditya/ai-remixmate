# RemixMate provision runbook (Phase 7)

Code on `hosted-app` is deploy ready after Phases 0 to 8. This runbook is the
**operator** path that unlocks the resume line *"deployed at [URL]"*. Until every
live verify box is green against real output, the honest claim stays
**Dockerized**.

Run the steps in order. Every variable name below is one the code reads; the
full list with comments lives in [`remixmate.env.example`](remixmate.env.example).

## 1. Accounts and hardware

1. Buy a Hetzner CX22 (~€4.5/mo) plus a ~60 GB volume (~€3/mo).
2. Create a Supabase project (free tier is enough for demo traffic).
3. Create a Cloudflare Pages project for the React frontend (free).

## 2. OS

Pick **Ubuntu 24.04** as the server image. `bootstrap.sh` installs
`python3.12` and `python3.12-venv` from apt and runs as root.

## 3. Catalog volume: format, mount, fstab

The service unit has `RequiresMountsFor=/mnt/catalog`, so `remixmate` will not
start until the volume is mounted there.

```bash
# as root; find the volume device (Hetzner names it scsi-0HC_Volume_<id>)
ls -l /dev/disk/by-id/
# format only a new, empty volume (this erases it):
mkfs.ext4 /dev/disk/by-id/scsi-0HC_Volume_<id>
mkdir -p /mnt/catalog
echo '/dev/disk/by-id/scsi-0HC_Volume_<id> /mnt/catalog ext4 defaults,nofail 0 2' >> /etc/fstab
mount -a
findmnt /mnt/catalog
```

The service itself cannot write to `/mnt/catalog`: `ProtectSystem=strict` leaves
only `ReadWritePaths=/opt/remixmate/data /opt/remixmate/outputs` writable, plus
`CacheDirectory=remixmate` (`/var/cache/remixmate`), where the unit points
`NUMBA_CACHE_DIR`, `XDG_CACHE_HOME` and `REMIXMATE_CACHE_DIR`; librosa's numba
JIT fails on the first mix without it. After
the catalog rsync (section 11) you may also switch the fstab option from
`defaults` to `ro` and run `mount -o remount,ro /mnt/catalog`.

## 4. Firewall

Open only SSH, HTTP and HTTPS. Uvicorn binds `127.0.0.1:8000`, so the API port is
never exposed; Caddy fronts it.

```bash
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
ufw status
```

## 5. DNS

- `api.<domain>`: A record to the VPS IP. If Cloudflare manages the zone, set
  this record to **DNS only** (grey cloud), not proxied, so Caddy can obtain its
  own certificate and the SSE stream on `/events/*` reaches the client unbuffered.
- `<domain>`: custom domain on the Cloudflare Pages project (section 10).

## 6. Supabase settings

In the Supabase dashboard:

1. **Site URL**: Authentication, URL Configuration, set to `https://<domain>`.
2. **Email confirmation**: Authentication, Providers, Email, keep "Confirm email"
   **on**.
3. **JWT signing key must be RS256.** The API verifies tokens with
   `algorithms=["RS256"]` against the project JWKS (`scripts/api/auth.py`); a
   token signed with any other algorithm is rejected.
4. Collect these values:

| Value | Goes to |
|---|---|
| Project URL `https://<ref>.supabase.co` | `SUPABASE_URL` (API), `VITE_SUPABASE_URL` (Pages) |
| anon (public) key | `VITE_SUPABASE_ANON_KEY` (Pages) |
| JWKS URL `https://<ref>.supabase.co/auth/v1/.well-known/jwks.json` | `SUPABASE_JWKS_URL` (API) |
| Issuer `https://<ref>.supabase.co/auth/v1` (optional check) | `SUPABASE_JWT_ISS` (API) |
| service role key (optional; enables `DELETE /account` user purge) | `SUPABASE_SERVICE_ROLE_KEY` (API only, never Pages) |

## 7. Postgres

Use **Supabase Postgres**, not a local server. In Supabase, Connect, copy the
**pooler** connection string, then change its scheme from `postgresql://` to
`postgresql+psycopg://` (the SQLAlchemy driver form the API needs):

```
DATABASE_URL=postgresql+psycopg://<user>:<password>@<pooler-host>:<port>/postgres
```

`backup.sh` strips `+psycopg` itself before calling `pg_dump`.

## 8. Deploy key and clone

The repo is private. Give the VPS a read only deploy key:

```bash
# as root on the VPS
ssh-keygen -t ed25519 -f /root/.ssh/remixmate_deploy -N ''
cat /root/.ssh/remixmate_deploy.pub
```

Add the public key in GitHub, repo Settings, Deploy keys (leave write access
off). Then clone to the path every script expects:

```bash
GIT_SSH_COMMAND='ssh -i /root/.ssh/remixmate_deploy' \
  git clone -b hosted-app --single-branch git@github.com:<owner>/<repo>.git /opt/remixmate
# keep the deploy key for later pulls
git -C /opt/remixmate config core.sshCommand 'ssh -i /root/.ssh/remixmate_deploy'
```

Clone `hosted-app`, not the default branch: `main` has no `app/deploy/`, so
section 9 would find no env example, Caddyfile or `bootstrap.sh`.
`bootstrap.sh` refuses to run until `/opt/remixmate/.git` exists.

## 9. Environment file, Caddyfile, bootstrap, Caddy reload

1. Fill the env file:

   ```bash
   cp /opt/remixmate/app/deploy/remixmate.env.example /etc/remixmate.env
   nano /etc/remixmate.env
   chmod 600 /etc/remixmate.env
   ```

   Set at least `REMIXMATE_AUTH_ENABLED=true`, `SUPABASE_URL`,
   `SUPABASE_JWKS_URL`, `ADMIN_EMAILS`, `DATABASE_URL` (section 7),
   `REMIXMATE_LIBRARY_DIR=/mnt/catalog`, `REMIXMATE_TRUST_PROXY=1`,
   `REMIXMATE_REQUIRE_CATALOG=1` and `REMIXMATE_FRONTEND_URL=https://<domain>`.
   Leave `ADMIN_ALLOW_UNVERIFIED`, `REMIXMATE_STEM_SPLIT_ENABLED` and
   `REMIXMATE_GENERATIVE_ENABLED` unset. `REMIXMATE_LICENSE_ENFORCE` gates
   ingest only; the API just reports it. Optional: `SUPABASE_JWT_ISS`,
   `SUPABASE_SERVICE_ROLE_KEY`, `REMIXMATE_DEVICE`,
   `REMIXMATE_MUSICBRAINZ_ENABLED`, `GENIUS_TOKEN`, the `REMIXMATE_SMTP_*`
   block with `REMIXMATE_DMCA_CONTACT` (or `REMIXMATE_SMTP_TO` as fallback).

2. Replace `YOUR_DOMAIN` in `/opt/remixmate/app/deploy/Caddyfile` with your
   domain. `bootstrap.sh` stops if the placeholder is still there.

3. Run the bootstrap:

   ```bash
   sudo bash /opt/remixmate/app/deploy/bootstrap.sh
   ```

   It is idempotent. It checks for root, installs packages, creates the
   `remixmate` system user (`useradd -r -M -d /opt/remixmate`), builds `.venv`
   with `python3.12`, gives the user write access to `data/` and `outputs/`
   only, installs the systemd unit, installs the Caddyfile to
   `/etc/caddy/Caddyfile` and runs `systemctl reload caddy`, restarts
   `remixmate`, and prints the `/health/ready` result. If `/etc/remixmate.env`
   is missing it installs the unit, prints what to do, and exits 1 without
   starting the service; fill the file and re-run.

   With `REMIXMATE_REQUIRE_CATALOG=1`, `/health/ready` returns 503 until the
   catalog is on `/mnt/catalog` (section 11). Recheck after the rsync:

   ```bash
   curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/health/ready
   ```

4. **Caddy reload** after any later Caddyfile edit:

   ```bash
   install -m 644 /opt/remixmate/app/deploy/Caddyfile /etc/caddy/Caddyfile
   systemctl reload caddy
   ```

Optional: `systemd-analyze verify /etc/systemd/system/remixmate.service`

## 10. Frontend (Cloudflare Pages)

Project settings:

- Root directory: `frontend`
- Build command: `npm ci && npm run build`
- Build output directory: `dist`

Environment variables (build time):

- `VITE_API_BASE=https://api.<domain>`
- `VITE_EVENTS_URL=https://api.<domain>/events/stream`
- `VITE_SUPABASE_URL=https://<ref>.supabase.co`
- `VITE_SUPABASE_ANON_KEY=<anon key from section 6>`
- `VITE_COOKIE_DOMAIN=<domain>`: sets the auth cookie's `Domain` so the SSE and
  file routes on `api.<domain>` receive it.

Attach `<domain>` as the Pages custom domain. It must match
`REMIXMATE_FRONTEND_URL` (or `REMIXMATE_CORS_ORIGINS`) on the API.

## 11. Catalog

Build the catalog on the ingest machine from the repo root, with the ingest env
so it writes to `catalog/` and never touches the personal library:

```bash
export REMIXMATE_LIBRARY_LIBRARY_DIR="$PWD/catalog"
export REMIXMATE_DATA_DIR="$PWD/catalog_data"
export REMIXMATE_LICENSE_ENFORCE=1
python -m scripts.curate_catalog            # dry run
python -m scripts.curate_catalog --execute  # ingest
```

Curate ~150 CC tracks (ccMixter + Internet Archive). Demucs runs at ingest
(~35 GB). Each track folder should carry `license.json`, its stems,
`meta.json` and `analysis.json`.

Sync **from `catalog/`, never from `library/`** (`library/` is the personal
library, not licensed for hosting). The trailing slash copies the folder
contents:

```bash
rsync -av --delete catalog/ root@<vps-ip>:/mnt/catalog/
# then on the VPS, so the remixmate user can read it:
chmod -R a+rX /mnt/catalog
```

The catalog is **not** part of the git deploy.

## 12. Backups: rclone and cron

[`backup.sh`](backup.sh) dumps the database, tars `data/` (audit log and model
JSON), tars `outputs/` too when `BACKUP_OUTPUTS=1`, uploads with rclone when
`RCLONE_REMOTE` is set, and prunes local files older than 14 days. It reads only
`DATABASE_URL`, `BACKUP_DIR` (default `/var/backups/remixmate`), `RCLONE_REMOTE`
and `BACKUP_OUTPUTS` from `/etc/remixmate.env`, line by line, without sourcing it.

1. **rclone** (object storage such as Backblaze or R2):

   ```bash
   apt install -y rclone
   rclone config          # as root, create a remote, e.g. named b2
   ```

   Add `RCLONE_REMOTE=<remote>:<bucket>/<path>` to `/etc/remixmate.env`.

2. **cron**: the script is committed executable (mode 100755), so root's
   crontab calls it directly (`crontab -e` as root):

   ```cron
   0 2 * * * /opt/remixmate/app/deploy/backup.sh >> /var/log/remixmate-backup.log 2>&1
   ```

   Run it once by hand first and read the log.

## 13. Live verify (must all pass)

`live_verify` needs two access tokens from two **different** confirmed users.
Sign up both through the site, confirm each email, then fetch each token from
Supabase:

```bash
curl -sS -X POST 'https://<ref>.supabase.co/auth/v1/token?grant_type=password' \
  -H 'apikey: <anon key>' -H 'Content-Type: application/json' \
  -d '{"email":"<user a>","password":"<password>"}'
# copy access_token into a file; repeat for user B
```

Pass the tokens by file or env var so JWTs never land in shell history:

```bash
cd /opt/remixmate
.venv/bin/python -m scripts.system.live_verify \
  --base-url https://api.<domain> \
  --token-a-file /root/token_a --token-b-file /root/token_b \
  --song-a "<catalog_song_name>" \
  --song-b "<catalog_song_name>"
rm /root/token_a /root/token_b
```

`LIVE_VERIFY_TOKEN_A` and `LIVE_VERIFY_TOKEN_B` work in place of the file flags.
All harness checks must pass (exit 0), including cross tenant isolation.

Manual checklist:

- [ ] Sign up, log in, render a mix end to end; download includes `CREDITS.txt`
- [ ] `systemctl restart remixmate` drains in flight jobs; `/health/ready` returns 200
- [ ] `pg_dump` dumps and restores clean into a scratch DB
- [ ] **Cross tenant:** harness exit 0 (and second account cannot list first account's mixes)
- [ ] Median mix render time on the CX22 is acceptable to show a recruiter

Only after every box is ticked: update the resume from "Dockerized" to
"deployed at [URL]".

## 14. CI

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs ruff, the
pytest safe subset, and frontend `tsc` / vitest / vite build on PRs. Leave
`pages.yml` and `codeql.yml` disabled unless you intentionally re-enable them.
