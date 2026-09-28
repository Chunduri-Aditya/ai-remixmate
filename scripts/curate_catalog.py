#!/usr/bin/env python3
"""
scripts/curate_catalog.py — Batch catalogue-curation driver for AI RemixMate.

Loads a manifest of search queries across ccMixter and Internet Archive, classifies
licence terms, estimates storage, and optionally ingests publishable tracks.

Usage:
  python -m scripts.curate_catalog                  # dry-run (estimate only)
  python -m scripts.curate_catalog --execute        # actually ingest
  python -m scripts.curate_catalog --limit-entries 5  # test on first 5 entries
  python -m scripts.curate_catalog --manifest custom.json

The manifest is a JSON array of objects:
  {"source": "ccmixter"|"archive", "query": "search terms", "pick": 5, "notes": "optional"}

Set REMIXMATE_LICENSE_ENFORCE=1 to reject non-CC/PD tracks during ingest.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from scripts.core.license import classify_license, is_publishable, license_enforce_enabled
from scripts.core.paths import DATA_DIR
from scripts.download import ingest_track, search_catalog

log = logging.getLogger(__name__)

# Estimate 0.2 GB per accepted track with stems (4 × 44.1k mono WAV, ~3-4 min avg)
_GB_PER_TRACK = 0.2

_STATE_FILE = DATA_DIR / "catalog_curate_state.json"


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load the manifest JSON array."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"manifest must be a JSON array, got {type(data).__name__}")
    return data


def load_state() -> Dict[str, bool]:
    """Load resume state (completed query keys)."""
    if not _STATE_FILE.exists():
        return {}
    with open(_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, bool]) -> None:
    """Persist resume state."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def query_key(entry: Dict[str, Any]) -> str:
    """Unique key for a manifest entry."""
    return f"{entry['source']}:{entry['query']}"


def curate_catalog(
    manifest_path: Path,
    dry_run: bool = True,
    limit_entries: int | None = None,
) -> int:
    """
    Main curation loop.
    Returns 0 on success (or dry-run completion), 1 if --execute and zero successes.
    """
    manifest = load_manifest(manifest_path)
    state = load_state() if not dry_run else {}

    if limit_entries is not None:
        manifest = manifest[:limit_entries]
        log.info("Limiting to first %d entries", limit_entries)

    if dry_run:
        log.info("DRY RUN — estimating publishable tracks only (no ingest)")
    else:
        log.info("EXECUTE mode — will ingest publishable tracks")
        if not license_enforce_enabled():
            log.warning(
                "REMIXMATE_LICENSE_ENFORCE is not set — all tracks will be ingested "
                "regardless of licence publishability"
            )

    total_candidates = 0
    publishable_count = 0
    estimated_tracks = 0
    ingest_success_count = 0

    for idx, entry in enumerate(manifest, 1):
        source = entry["source"]
        query = entry["query"]
        pick = entry.get("pick", 5)
        notes = entry.get("notes", "")
        key = query_key(entry)

        if not dry_run and state.get(key):
            log.info("[%d/%d] Skipping (already completed): %s / %r", idx, len(manifest), source, query)
            continue

        log.info("[%d/%d] %s / %r  (pick %d)%s", idx, len(manifest), source, query, pick,
                 f"  — {notes}" if notes else "")

        try:
            candidates = search_catalog(source, query, limit=pick * 2)
        except Exception as exc:
            log.error("Search failed for %r: %s", query, exc)
            continue

        if not candidates:
            log.info("  → no results")
            continue

        log.info("  → found %d candidates", len(candidates))

        picked = 0
        entry_publishable = 0
        for cand in candidates:
            total_candidates += 1
            info = classify_license(
                cand.source, license_url=cand.license_url, license_str=cand.license_str
            )
            publishable = is_publishable(info)
            status = "✓ publishable" if publishable else "✗ not publishable"

            log.info(
                "    %s — %s  [%s] %s",
                cand.artist or "?", cand.title, info.license_type.value, status
            )

            if publishable:
                publishable_count += 1
                entry_publishable += 1

            if not dry_run and publishable and picked < pick:
                try:
                    result = ingest_track(cand, separate=True, auto_analyze=True)
                    if result.success:
                        ingest_success_count += 1
                        picked += 1
                        log.info("      ✓ ingested: library/%s/", result.name)
                    else:
                        log.warning("      ✗ ingest failed: %s", result.error)
                except Exception as exc:
                    log.error("      ✗ ingest exception: %s", exc)

            if picked >= pick:
                break

        estimated_tracks += min(pick, entry_publishable)

        # an entry is done only once at least one track ingested
        if not dry_run and picked > 0:
            state[key] = True
            save_state(state)

    estimated_gb = estimated_tracks * _GB_PER_TRACK
    log.info("")
    log.info("=" * 60)
    log.info("Total candidates examined: %d", total_candidates)
    log.info("Publishable tracks: %d", publishable_count)
    log.info("Estimated storage (0.2 GB per track with stems): %.1f GB", estimated_gb)

    if dry_run:
        log.info("Dry-run complete. Use --execute to ingest.")
        return 0

    log.info("Ingest successes: %d", ingest_success_count)
    if ingest_success_count == 0:
        log.error("No tracks successfully ingested")
        return 1

    return 0


def _build_parser() -> argparse.ArgumentParser:
    default_manifest = Path(__file__).parent / "catalog_manifest.json"
    p = argparse.ArgumentParser(
        prog="python -m scripts.curate_catalog",
        description="Batch catalogue-curation driver (CC / public-domain sources)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest,
        help=f"path to manifest JSON (default: {default_manifest})"
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="actually ingest publishable tracks (default is dry-run)"
    )
    p.add_argument(
        "--limit-entries",
        type=int,
        metavar="N",
        help="test on first N manifest entries only"
    )
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        return 1

    dry_run = not args.execute
    return curate_catalog(args.manifest, dry_run=dry_run, limit_entries=args.limit_entries)


if __name__ == "__main__":
    raise SystemExit(main())
