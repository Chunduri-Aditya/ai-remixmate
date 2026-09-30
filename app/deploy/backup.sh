#!/usr/bin/env bash
set -euo pipefail

# RemixMate backup: database dump plus data/ (and outputs/ with BACKUP_OUTPUTS=1).
# Requires DATABASE_URL, from the environment or /etc/remixmate.env.
# Dumps and archives older than 14 days are pruned.

# Read the env file line by line (KEY=value) instead of sourcing it, so the
# file is never executed as shell. Only the keys this script uses are taken,
# and a value already set in the environment wins.
ENV_FILE="/etc/remixmate.env"
if [ -f "$ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      ''|'#'*) continue ;;
    esac
    key="${line%%=*}"
    [ "$key" = "$line" ] && continue
    value="${line#*=}"
    case "$key" in
      DATABASE_URL|BACKUP_DIR|RCLONE_REMOTE|BACKUP_OUTPUTS) ;;
      *) continue ;;
    esac
    if [[ "$value" == \"*\" || "$value" == \'*\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    if [ -z "${!key:-}" ]; then
      export "$key=$value"
    fi
  done < "$ENV_FILE"
fi

if [ -z "${DATABASE_URL:-}" ]; then
  echo "ERROR: DATABASE_URL not set" >&2
  echo "Either export it or ensure /etc/remixmate.env exists and contains it" >&2
  exit 1
fi

BACKUP_DIR="${BACKUP_DIR:-/var/backups/remixmate}"
APP_DIR="${APP_DIR:-/opt/remixmate}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BACKUP_FILE="$BACKUP_DIR/remixmate-$TIMESTAMP.sql.gz"
DATA_ARCHIVE="$BACKUP_DIR/remixmate-data-$TIMESTAMP.tar.gz"
OUTPUTS_ARCHIVE="$BACKUP_DIR/remixmate-outputs-$TIMESTAMP.tar.gz"

# pg_dump rejects the SQLAlchemy driver form postgresql+psycopg://
PG_URL="${DATABASE_URL/postgresql+psycopg:/postgresql:}"

echo "==> Creating backup directory..."
mkdir -p "$BACKUP_DIR"

echo "==> Backing up database to $BACKUP_FILE"
pg_dump "$PG_URL" | gzip > "$BACKUP_FILE"

echo "✓  Backup created: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"

echo "==> Archiving data/ to $DATA_ARCHIVE"
tar -czf "$DATA_ARCHIVE" -C "$APP_DIR" data
UPLOADS=("$BACKUP_FILE" "$DATA_ARCHIVE")

if [ "${BACKUP_OUTPUTS:-0}" = "1" ]; then
  echo "==> Archiving outputs/ to $OUTPUTS_ARCHIVE"
  tar -czf "$OUTPUTS_ARCHIVE" -C "$APP_DIR" outputs
  UPLOADS+=("$OUTPUTS_ARCHIVE")
fi

if [ -n "${RCLONE_REMOTE:-}" ]; then
  echo "==> Uploading to $RCLONE_REMOTE"
  for f in "${UPLOADS[@]}"; do
    rclone copy "$f" "$RCLONE_REMOTE"
  done
  echo "✓  Uploaded to remote storage"
fi

echo "==> Pruning local backups older than 14 days"
find "$BACKUP_DIR" -maxdepth 1 -type f -name 'remixmate-*' -mtime +14 -delete

echo ""
echo "Backup complete!"
