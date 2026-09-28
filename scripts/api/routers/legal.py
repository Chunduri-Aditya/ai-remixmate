"""
scripts/api/routers/legal.py — Public legal pages + DMCA intake (Phase 8).

Mounted WITHOUT ``require_user`` so unauthenticated visitors can read terms
and file a DMCA notice. Does not read per-user mixes/crates/jobs.
"""

from __future__ import annotations

import logging
import os
import smtplib
import time
from collections import defaultdict
from email.message import EmailMessage
from typing import DefaultDict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from scripts.core import legal_store
from scripts.core.config import cfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/legal", tags=["legal"])

# Simple in-memory rate limit: max N POSTs per IP per window.
_DMCA_WINDOW_SEC = 3600
_DMCA_MAX_PER_WINDOW = 5
_dmca_hits: DefaultDict[str, List[float]] = defaultdict(list)

_PAGES = {
    "terms": """AI RemixMate — Terms of Service

Not affiliated with ccMixter, Free Music Archive, Internet Archive, Musopen,
Spotify, or any rights holder.

You may use RemixMate to render mixes from the hosted Creative Commons catalog.
Mixes carry attribution in CREDITS.txt; you must preserve that attribution when
you redistribute a mix.

Account data we store: email (via Supabase Auth), rendered mixes, crates,
favorites, tags, feedback events, and job metadata. You may export or delete
your account at any time (see Privacy and the account API).

Acceptable use: do not upload or claim rights you do not hold; do not abuse the
API; do not attempt to circumvent license gates. We may flag or remove accounts
that violate these terms.

Contact for legal notices: {contact}
""",
    "privacy": """AI RemixMate — Privacy Policy

Data stored
- Email and auth identifiers via Supabase
- Per-user rendered mixes under outputs/<user_id>/
- Crates, favorites, tags, feedback events, job history

We do not sell personal data. Auth tokens are verified via Supabase JWKS.

Audit log
- Job starts and completions are recorded in an audit log under your user id
- DELETE /account removes your lines from the audit log

Retention
- A rendered mix is deleted after {mix_max_age_days} days
- At most {mix_max_count} mixes are kept per user; the oldest are removed first

Your rights
- GET /account/export — download a zip of manifests, crates, and favorites
- DELETE /account — purge your rows, files, and (when configured) Supabase user

Hosted demos may use cloud providers as disclosed in the deploy env.
""",
    "dmca": """AI RemixMate — DMCA / Takedown

To report alleged copyright infringement, POST /legal/dmca with:
  reporter_name, reporter_email, copyright_work, infringing_url, description

Reports are stored and (when REMIXMATE_SMTP_* is configured) emailed to the
operator. Agent contact: {contact}.

Not affiliated with ccMixter, Free Music Archive, Internet Archive, Musopen,
Spotify, or any rights holder.
""",
    "acceptable-use": """AI RemixMate — Acceptable Use

Allowed: personal and commercial use of mixes rendered from catalog tracks that
carry a remix-compatible Creative Commons license, with CREDITS.txt attribution
preserved.

Prohibited: uploading third-party works you do not control; stripping
attribution; harassing other users; attempting cross-tenant data access;
automated scraping that degrades service.

Violations may result in account flagging (403 on API) or mix takedown.
""",
}


class DmcaRequest(BaseModel):
    reporter_name: str = Field(..., min_length=1, max_length=200)
    reporter_email: str = Field(..., min_length=3, max_length=320)
    copyright_work: str = Field(..., min_length=1, max_length=2000)
    infringing_url: str = Field(..., min_length=1, max_length=2000)
    description: str = Field("", max_length=5000)


def _client_ip(request: Request) -> str:
    """Limiter key: the socket peer, or with REMIXMATE_TRUST_PROXY=1 the last
    X-Forwarded-For entry (the one the reverse proxy appends; earlier entries
    are client supplied and spoofable)."""
    if os.environ.get("REMIXMATE_TRUST_PROXY") == "1":
        forwarded = request.headers.get("x-forwarded-for", "")
        last = forwarded.split(",")[-1].strip()
        if last:
            return last
    if request.client:
        return request.client.host
    return "unknown"


def _rate_limit_dmca(ip: str) -> None:
    now = time.time()
    window_start = now - _DMCA_WINDOW_SEC
    # Prune every key on every call so the dict cannot grow without bound.
    for key in list(_dmca_hits):
        fresh = [t for t in _dmca_hits[key] if t >= window_start]
        if fresh:
            _dmca_hits[key] = fresh
        else:
            del _dmca_hits[key]
    hits = list(_dmca_hits.get(ip, []))
    if len(hits) >= _DMCA_MAX_PER_WINDOW:
        raise HTTPException(429, detail="too many DMCA reports — try later")
    hits.append(now)
    _dmca_hits[ip] = hits


def _send_dmca_email(report_id: str, body: DmcaRequest) -> bool:
    """Best-effort SMTP. Returns True if sent; False if stubbed/unavailable.

    Flagged: email is stubbed behind REMIXMATE_SMTP_* — when unset we log and
    return False without raising (intake still persists the row).
    """
    host = os.environ.get("REMIXMATE_SMTP_HOST", "").strip()
    if not host:
        logger.info(
            "DMCA email stubbed (REMIXMATE_SMTP_HOST unset) report_id=%s", report_id
        )
        return False

    port = int(os.environ.get("REMIXMATE_SMTP_PORT", "587") or "587")
    user = os.environ.get("REMIXMATE_SMTP_USER", "").strip()
    password = os.environ.get("REMIXMATE_SMTP_PASSWORD", "").strip()
    mail_from = os.environ.get("REMIXMATE_SMTP_FROM", user or "noreply@localhost")
    mail_to = (
        os.environ.get("REMIXMATE_DMCA_CONTACT", "").strip()
        or os.environ.get("REMIXMATE_SMTP_TO", "").strip()
        or mail_from
    )

    msg = EmailMessage()
    msg["Subject"] = f"[RemixMate DMCA] {report_id}"
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(
        f"report_id: {report_id}\n"
        f"reporter: {body.reporter_name} <{body.reporter_email}>\n"
        f"work: {body.copyright_work}\n"
        f"url: {body.infringing_url}\n\n"
        f"{body.description}\n"
    )

    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            if user:
                smtp.login(user, password)
            smtp.send_message(msg)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("DMCA email failed report_id=%s: %s", report_id, exc)
        return False


@router.get("/{page}", response_class=PlainTextResponse)
def legal_page(page: str):
    text = _PAGES.get(page)
    if text is None:
        raise HTTPException(404, detail="unknown legal page")
    # Filled per request so an env change shows without a restart, and the
    # retention numbers come from the config the sweep enforces.
    contact = os.environ.get("REMIXMATE_DMCA_CONTACT", "").strip() or "contact not configured"
    text = (
        text.replace("{contact}", contact)
        .replace("{mix_max_age_days}", str(int(cfg.retention.mix_max_age_days)))
        .replace("{mix_max_count}", str(int(cfg.retention.mix_max_count)))
    )
    return PlainTextResponse(text.strip() + "\n")


@router.post("/dmca")
def submit_dmca(body: DmcaRequest, request: Request):
    _rate_limit_dmca(_client_ip(request))
    report_id = legal_store.create_dmca_report(
        reporter_name=body.reporter_name,
        reporter_email=str(body.reporter_email),
        copyright_work=body.copyright_work,
        infringing_url=body.infringing_url,
        description=body.description,
    )
    emailed = _send_dmca_email(report_id, body)
    return {"report_id": report_id, "email_sent": emailed}
