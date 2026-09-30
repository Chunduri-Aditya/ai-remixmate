# Security Policy

## Supported code

Only the latest code is maintained. Reproduce the issue on the current commit before reporting, and name that commit hash in the report.

## Reporting a vulnerability

Email is the only channel: **chunduriaditya2@gmail.com**, with `SECURITY: ai-remixmate` in the subject line.

The repository is private, so GitHub private vulnerability reporting is not available for it. Do not put vulnerability details in a public issue, a public fork, or any other public place.

## What to include

- The commit hash you tested
- Whether `REMIXMATE_AUTH_ENABLED` was on or off
- The exact request, command or input file
- What you expected and what happened
- A proof of concept, if you have one

## What to expect

One maintainer reads the security inbox. When a fix is needed, the disclosure is agreed with you before anything is made public.

## How authentication is gated

Authentication is controlled by the `REMIXMATE_AUTH_ENABLED` environment variable, read on every request (`scripts/api/auth.py`).

**Auth off (the default).** Every request runs as the user `local` and no token is needed. This mode is for one person on a machine they control; `./start.sh` binds the API to `127.0.0.1` unless `--host` says otherwise.

**Auth on.** A request carries a Supabase access token, taken from the `Authorization: Bearer` header first and then from the `sb-access-token` cookie (renamed with `SUPABASE_COOKIE_NAME`), which the EventSource stream needs because it cannot send headers. The cookie authenticates only `GET` and `HEAD` (`_COOKIE_METHODS` in `scripts/api/auth.py`), so a mutating request needs the Bearer header, which a cross site form cannot set. The token is verified as RS256 against the JWKS at `SUPABASE_JWKS_URL`, with audience `authenticated` by default (`SUPABASE_JWT_AUD`), an optional issuer pin (`SUPABASE_JWT_ISS`), required `exp` and `sub` claims, and a `role` claim that must be `authenticated`, so anon and service role keys are refused. The middleware pins the token's `sub` as the acting user for the request, and background jobs run again as the user stamped on the job.

Routes sit in four tiers:

| Tier | Dependency | Routes | Refusal |
|---|---|---|---|
| Open | none | `/health`, `/health/live`, `/health/ready`, `/features`, `/legal/{page}`, `POST /legal/dmca`, `/`, `/app`, `/docs`, `/redoc` | none |
| User | `require_user` | everything mounted through `scripts/api/routes.py` | 401 without a valid token when auth is on; 403 for a flagged account in either mode |
| Operator | `require_operator` | catalog mutating routes: library delete, prune, evict, initialize, scan and calibrate, index rebuilds, analysis jobs over the shared library, stem split and compress | passes as `local` when auth is off; requires admin when auth is on |
| Admin | `require_admin` | `/admin/*` | always 403 when auth is off; when on, the token's `email` claim must be listed in `ADMIN_EMAILS` and verified (unless `ADMIN_ALLOW_UNVERIFIED=1`) |

Further gates on top of the tiers:

- Stem split routes return 503 unless `REMIXMATE_STEM_SPLIT_ENABLED` is truthy, and generative routes return 503 unless `REMIXMATE_GENERATIVE_ENABLED` is truthy.
- One user may hold at most 4 active background jobs (`_MAX_ACTIVE_JOBS` in `scripts/api/jobs.py`); the next submit gets 429.
- Per user output lives under `outputs/<user_id>/`, and the stores filter by the acting user.

An open route must never read per user state, because with auth on an unauthenticated request reaches it as `local`.

## Scope

In scope:

- Reaching a `require_user`, `require_operator` or `require_admin` route without the credential it requires while auth is on
- Cross tenant access: one user reading, changing or deleting another user's mixes, crates, jobs, feedback or outputs
- An open route that returns or changes per user data
- Getting past the flagged account block or the per user active job cap
- Arbitrary code execution through any request or input file
- Path traversal in library, stem or output paths
- Secrets, tokens or credentials exposed in logs or responses

Out of scope:

- Vulnerabilities in third party dependencies such as librosa, Demucs, FastAPI, PyJWT or Supabase; report those upstream
- Anything that needs auth off: in that mode every caller is `local` by design, so running it on a network you do not control is a deployment mistake, not a flaw in the code
- Issues that need physical access to the machine running the server
- Denial of service through expensive jobs that stay within the per user cap; the API is not hardened against resource exhaustion
