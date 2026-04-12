"""
scripts/core/license.py — Licence classification and tracking for AI RemixMate.

Every song that enters the library gets a LicenseInfo record saved alongside it
in  library/<name>/license.json .

The module answers three questions the user (and the app) care about:
  1. Can I use this track commercially?
  2. Do I need to attribute the artist?
  3. Where did this track come from, and can I legally redistribute remixes?

Supported sources:
  • YouTube / YouTube Music  — YouTube ToS, personal use only
  • Jamendo                  — Creative Commons (various flavours)
  • SoundCloud               — varies per track; best-effort CC parsing
  • Local / Unknown          — user-provided; assumed unknown

Usage:
  from scripts.core.license import classify_license, license_warning, LicenseInfo

  info = classify_license(source="jamendo", license_url="https://creativecommons.org/licenses/by/4.0/")
  print(license_warning(info, "Some Song"))
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Licence type enum
# ---------------------------------------------------------------------------

class LicenseType(str, Enum):
    CC0          = "cc0"           # Public Domain — use freely, no attribution needed
    CC_BY        = "cc_by"         # Attribution required, otherwise free
    CC_BY_SA     = "cc_by_sa"      # Attribution + ShareAlike (remixes must share-alike)
    CC_BY_ND     = "cc_by_nd"      # Attribution + NoDerivatives — remixes NOT allowed
    CC_BY_NC     = "cc_by_nc"      # Attribution + NonCommercial
    CC_BY_NC_SA  = "cc_by_nc_sa"   # Attribution + NonCommercial + ShareAlike
    CC_BY_NC_ND  = "cc_by_nc_nd"   # Most restrictive CC — NC + NoDerivatives
    YOUTUBE_TOS  = "youtube_tos"   # YouTube Terms of Service — personal use only
    PROPRIETARY  = "proprietary"   # Known copyright, all rights reserved
    UNKNOWN      = "unknown"       # Could not determine; treat conservatively


# ---------------------------------------------------------------------------
# LicenseInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class LicenseInfo:
    """Normalised licence record stored per song."""

    license_type: LicenseType
    source: str                         # "youtube", "jamendo", "soundcloud", "local", etc.

    # Derived rights (populated by classify_license)
    commercial_ok: bool = False         # safe for commercial projects?
    attribution_required: bool = True   # must credit artist?
    derivatives_ok: bool = True         # remixing/editing allowed?
    redistribution_ok: bool = False     # sharing the track (or remix) publicly OK?
    share_alike: bool = False           # remixes must use the same licence?

    license_url: Optional[str] = None   # canonical URL for the licence
    notes: str = ""                     # human-readable extra info


# ---------------------------------------------------------------------------
# CC URL parser
# ---------------------------------------------------------------------------

# Maps CC identifier slugs (from URL) → LicenseType
_CC_URL_MAP: dict[str, LicenseType] = {
    "cc0":          LicenseType.CC0,
    "zero":         LicenseType.CC0,
    "by/":          LicenseType.CC_BY,
    "by-sa":        LicenseType.CC_BY_SA,
    "by-nd":        LicenseType.CC_BY_ND,
    "by-nc/":       LicenseType.CC_BY_NC,
    "by-nc-sa":     LicenseType.CC_BY_NC_SA,
    "by-nc-nd":     LicenseType.CC_BY_NC_ND,
}


def parse_cc_url(url: str) -> Optional[LicenseType]:
    """
    Try to extract a LicenseType from a Creative Commons licence URL.

    Handles formats like:
      https://creativecommons.org/licenses/by/4.0/
      https://creativecommons.org/licenses/by-nc-sa/3.0/
      https://creativecommons.org/publicdomain/zero/1.0/
    """
    if not url:
        return None
    url_lower = url.lower()

    # Public domain / CC0
    if "publicdomain" in url_lower or "cc0" in url_lower or "/zero/" in url_lower:
        return LicenseType.CC0

    if "creativecommons.org" not in url_lower and "cc.org" not in url_lower:
        return None

    # Match the licence slug in the URL path
    for slug, lt in sorted(_CC_URL_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if slug in url_lower:
            return lt

    return None


# ---------------------------------------------------------------------------
# Rights derivation table
# ---------------------------------------------------------------------------

# (commercial_ok, attribution_required, derivatives_ok, redistribution_ok, share_alike)
_RIGHTS: dict[LicenseType, tuple[bool, bool, bool, bool, bool]] = {
    LicenseType.CC0:         (True,  False, True,  True,  False),
    LicenseType.CC_BY:       (True,  True,  True,  True,  False),
    LicenseType.CC_BY_SA:    (True,  True,  True,  True,  True),
    LicenseType.CC_BY_ND:    (True,  True,  False, True,  False),
    LicenseType.CC_BY_NC:    (False, True,  True,  False, False),
    LicenseType.CC_BY_NC_SA: (False, True,  True,  False, True),
    LicenseType.CC_BY_NC_ND: (False, True,  False, False, False),
    LicenseType.YOUTUBE_TOS: (False, True,  False, False, False),
    LicenseType.PROPRIETARY: (False, True,  False, False, False),
    LicenseType.UNKNOWN:     (False, True,  False, False, False),
}


def _rights_from_type(lt: LicenseType) -> tuple[bool, bool, bool, bool, bool]:
    return _RIGHTS.get(lt, _RIGHTS[LicenseType.UNKNOWN])


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_license(
    source: str,
    license_url: Optional[str] = None,
    license_str: Optional[str] = None,
) -> LicenseInfo:
    """
    Classify a track's licence based on its source and any licence metadata.

    Parameters
    ----------
    source : str
        Where the track came from.  Known values: "youtube", "ytmusicapi",
        "yt-dlp", "jamendo", "soundcloud", "local".
    license_url : str, optional
        Raw licence URL returned by the metadata (e.g. Jamendo CC URL).
    license_str : str, optional
        Raw licence string from yt-dlp info JSON (e.g. "Creative Commons").

    Returns
    -------
    LicenseInfo with all rights fields populated.
    """
    src = source.lower().strip()

    # --- YouTube family ---
    if src in ("youtube", "ytmusicapi", "yt-dlp", "yt_dlp"):
        # Check if the video is itself CC-BY (some YT uploads use CC-BY)
        cc_type = None
        if license_url:
            cc_type = parse_cc_url(license_url)
        if license_str and "creative commons" in license_str.lower():
            # YT-DLP reports "Creative Commons Attribution licence (reuse allowed)"
            cc_type = cc_type or LicenseType.CC_BY

        if cc_type and cc_type != LicenseType.UNKNOWN:
            lt = cc_type
            notes = f"YouTube video with CC licence: {license_url or license_str}"
        else:
            lt = LicenseType.YOUTUBE_TOS
            notes = (
                "Downloaded from YouTube. Personal / portfolio use only. "
                "Commercial use requires licensing from the rights holder. "
                "For legal alternatives use --source jamendo."
            )

    # --- Jamendo (CC library) ---
    elif src == "jamendo":
        cc_type = parse_cc_url(license_url or "") if license_url else None
        lt = cc_type if cc_type else LicenseType.CC_BY  # Jamendo default is CC-BY
        notes = f"Jamendo track — {license_url or 'CC licence'}"

    # --- SoundCloud ---
    elif src == "soundcloud":
        cc_type = parse_cc_url(license_url or "") if license_url else None
        lt = cc_type if cc_type else LicenseType.UNKNOWN
        notes = "SoundCloud track — licence varies per upload."

    # --- User-provided local file ---
    elif src == "local":
        lt = LicenseType.UNKNOWN
        notes = "User-provided file — licence unknown; ensure you have rights to use."

    # --- Anything else ---
    else:
        cc_type = parse_cc_url(license_url or "") if license_url else None
        lt = cc_type if cc_type else LicenseType.UNKNOWN
        notes = f"Source: {source}. Licence could not be determined automatically."

    commercial, attribution, derivatives, redistribution, share_alike = _rights_from_type(lt)

    return LicenseInfo(
        license_type=lt,
        source=source,
        commercial_ok=commercial,
        attribution_required=attribution,
        derivatives_ok=derivatives,
        redistribution_ok=redistribution,
        share_alike=share_alike,
        license_url=license_url,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Human-readable warning / summary
# ---------------------------------------------------------------------------

def license_warning(info: LicenseInfo, track_name: str = "") -> str:
    """
    Return a human-readable warning string for a LicenseInfo.
    Returns an empty string if the licence is unrestricted (CC0 / CC-BY).
    """
    label = f"'{track_name}' " if track_name else ""
    lt = info.license_type

    if lt == LicenseType.CC0:
        return ""  # no warning needed

    if lt == LicenseType.CC_BY and info.commercial_ok:
        return (
            f"ℹ️  {label}is CC-BY — free to use commercially "
            f"but attribution to the artist is required."
        )

    if lt == LicenseType.YOUTUBE_TOS:
        return (
            f"⚠️  {label}was downloaded from YouTube.\n"
            f"   Personal / educational use is generally fine.\n"
            f"   Publishing or monetising remixes may violate YouTube's Terms of Service.\n"
            f"   For commercial projects, source from Jamendo (--source jamendo) instead."
        )

    if lt in (LicenseType.CC_BY_NC, LicenseType.CC_BY_NC_SA, LicenseType.CC_BY_NC_ND):
        return (
            f"⚠️  {label}is {lt.value.upper()} — NON-COMMERCIAL use only.\n"
            f"   You cannot use this remix in commercial projects.\n"
            f"   Attribution to the artist is required."
        )

    if lt == LicenseType.CC_BY_ND:
        return (
            f"⚠️  {label}is CC-BY-ND — No Derivatives.\n"
            f"   Remixing / modifying this track is NOT permitted under this licence.\n"
            f"   Consider using a different track."
        )

    if lt in (LicenseType.PROPRIETARY, LicenseType.UNKNOWN):
        return (
            f"🚨  {label}licence is {'proprietary/all-rights-reserved' if lt == LicenseType.PROPRIETARY else 'unknown'}.\n"
            f"   Do NOT publish or distribute remixes without permission from the rights holder.\n"
            f"   For legal music, use --source jamendo."
        )

    # Generic CC with share-alike
    if info.share_alike:
        return (
            f"ℹ️  {label}is {lt.value.upper()} — remixes must be released "
            f"under the same licence (ShareAlike).\n"
            f"   {'Commercial use is OK.' if info.commercial_ok else 'Non-commercial use only.'}"
        )

    return f"ℹ️  {label}({lt.value}) — review the licence before publishing."


# ---------------------------------------------------------------------------
# Persistence — save / load per-song
# ---------------------------------------------------------------------------

def save_license(song_dir: Path, info: LicenseInfo) -> None:
    """Save LicenseInfo as  <song_dir>/license.json ."""
    try:
        song_dir.mkdir(parents=True, exist_ok=True)
        data = asdict(info)
        data["license_type"] = info.license_type.value   # store string not enum
        with open(song_dir / "license.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning("Could not save license.json for %s: %s", song_dir.name, e)


def load_license(song_dir: Path) -> Optional[LicenseInfo]:
    """Load LicenseInfo from  <song_dir>/license.json . Returns None if absent."""
    path = song_dir / "license.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["license_type"] = LicenseType(data.get("license_type", "unknown"))
        return LicenseInfo(**data)
    except Exception as e:
        log.warning("Could not load license.json from %s: %s", song_dir.name, e)
        return None


def license_from_ytdlp_info(info_json_path: Path, source: str = "youtube") -> LicenseInfo:
    """
    Parse a yt-dlp info.json file and extract licence information.

    yt-dlp writes <song>/full.info.json when writeinfojson=True.
    """
    if not info_json_path.exists():
        return classify_license(source)
    try:
        with open(info_json_path, encoding="utf-8") as f:
            data = json.load(f)
        license_str = data.get("license") or ""
        # yt-dlp puts a human-readable URL in some fields; check several
        license_url = (
            data.get("license_url")
            or data.get("license_uri")
            or (license_str if license_str.startswith("http") else None)
        )
        return classify_license(source, license_url=license_url, license_str=license_str)
    except Exception as e:
        log.debug("Could not parse yt-dlp info JSON: %s", e)
        return classify_license(source)
