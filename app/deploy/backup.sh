#!/usr/bin/env bash
set -euo pipefail

# RemixMate database backup
# Requires DATABASE_URL environment variable.

# Source /etc/remixmate.env if it exists and DATABASE_URL is not set
if [ -z "${DATABASE_URL:-}" ] && [ -f /etc/remixmate.env ]; then
  # shellcheck disable=SC1091
  source /etc/remixmate.env
fi

if [ -z "${DATABASE_URL:-}" ]; then
  echo "ERROR: DATABASE_URL not set" >&2
  echo "Either export it or ensure /etc/remixmate.env exists and contains it" >&2
  exit 1
fi

BACKUP_DIR="${BACKUP_DIR:-/var/backups/remixmate}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BACKUP_FILE="$BACKUP_DIR/remixmate-$TIMESTAMP.sql.gz"

echo "==> Creating backup directory..."
mkdir -p "$BACKUP_DIR"

echo "==> Backing up database to $BACKUP_FILE"
pg_dump "$DATABASE_URL" | gzip > "$BACKUP_FILE"

echo "✓  Backup created: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"

if [ -n "${RCLONE_REMOTE:-}" ]; then
  echo "==> Uploading to $RCLONE_REMOTE"
  rclone copy "$BACKUP_FILE" "$RCLONE_REMOTE"
  echo "✓  Uploaded to remote storage"
fi

echo ""
echo "Backup complete!"
