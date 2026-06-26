#!/usr/bin/env python3
"""
scripts/purge_library.py — Remove library songs the engine cannot load.

A song is "unusable" when it has no resolvable source audio: no full.wav,
no full_enhanced.wav, and no Demucs stems — usually a download shell that
only ever got its metadata JSON written.

Usage:
    python -m scripts.purge_library            # dry run — list what would go
    python -m scripts.purge_library --apply    # actually delete them

Dry run is the default; deletion is irreversible.
"""

from __future__ import annotations

import argparse

from scripts.core.audio_source import find_unusable_songs, purge_unusable_songs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the unusable song folders (default: dry run).",
    )
    args = parser.parse_args()

    names = find_unusable_songs()
    if not names:
        print("✅  No unusable songs — every library entry has loadable audio.")
        return 0

    print(f"Found {len(names)} unusable song(s) (no full mix, no stems):")
    for n in names:
        print(f"  • {n}")

    if not args.apply:
        print(f"\nDry run — nothing deleted. Re-run with --apply to remove these {len(names)}.")
        return 0

    purge_unusable_songs(dry_run=False)
    print(f"\n🗑️  Deleted {len(names)} unusable song folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
