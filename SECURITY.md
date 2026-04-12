# Security Policy

## Supported versions

Only the current release is actively maintained. If you're on an older version, update to `main` first before reporting.

| Version | Supported |
|---|---|
| 0.2.x (current) | ✅ |
| 0.1.x | ❌ |
| < 0.1 | ❌ |

## Reporting a vulnerability

**Please don't open a public GitHub issue for security vulnerabilities.** If the issue is real, a public report gives anyone reading it the same information before it's fixed.

Instead, use one of these:

- **GitHub private advisory** (preferred) — [Report a vulnerability](../../security/advisories/new) via GitHub's private channel. I'll get notified privately and we can discuss it there.
- **Email** — chunduriaditya2@gmail.com. Put "SECURITY: ai-remixmate" in the subject line so it doesn't get buried.

## What to include

The more specific, the faster I can reproduce and fix it:

- What version / commit you're on
- What you did (exact command, request, config)
- What you expected vs. what happened
- Whether you have a proof of concept

## What to expect

This is a personal project with one maintainer. I'll do my best to respond within **7 days** and keep you updated on where things stand. If a fix is needed, I'll coordinate with you on disclosure timing before making anything public.

## Scope

Things that are in scope:

- Arbitrary code execution via any project input
- Path traversal in file handling (library paths, output paths)
- Authentication bypass if/when auth is added
- Server-side request forgery via the download or Spotify endpoints
- Secrets or credentials exposed in logs or responses

Things that are out of scope for this project:

- Vulnerabilities in third-party dependencies (report those upstream — librosa, Demucs, FastAPI, yt-dlp, etc.)
- Issues that require physical access to the machine running the server
- The download integrations themselves (yt-dlp, Spotify) — those services set their own terms
- Denial of service via large audio files (the `max_active_jobs` cap exists, but this isn't a hardened production system)

## Note on the download integrations

AI RemixMate uses `yt-dlp` and the Spotify API for personal use. These integrations are not security vulnerabilities in this project — they're third-party libraries with their own policies and maintainers. If you find a security issue in `yt-dlp` or `spotipy`, report it to those projects directly.
