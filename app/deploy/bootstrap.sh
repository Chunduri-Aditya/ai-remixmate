#!/usr/bin/env bash
set -euo pipefail

# RemixMate VPS bootstrap (idempotent)
# Mirrors PROVISION.md §2. Run as root on Debian/Ubuntu.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
  echo "ERROR: run this script as root (sudo $0)" >&2
  exit 1
fi

echo "==> Checking repository..."
if [ ! -d /opt/remixmate/.git ]; then
  cat <<EOF

Repository not found at /opt/remixmate/.git

Please clone the repository first:
  export REMIXMATE_REPO_URL=<your-repo-url>
  git clone \$REMIXMATE_REPO_URL /opt/remixmate

Then re-run this script.
EOF
  exit 1
fi

echo "==> Installing system packages..."
apt update
apt install -y ffmpeg libsndfile1 caddy postgresql curl python3.12 python3.12-venv

echo "==> Creating system user remixmate..."
if ! id remixmate &>/dev/null; then
  useradd -r -M -d /opt/remixmate remixmate
  echo "    Created user remixmate"
else
  echo "    User remixmate already exists"
fi

echo "==> Setting up repository..."
cd /opt/remixmate

echo "    Checking out hosted-app branch..."
git checkout hosted-app

echo "    Creating virtual environment..."
if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi

echo "    Installing Python dependencies..."
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/pip install --quiet -e .
echo "    (Skipped [generative] and [wordplay] extras)"

echo "    Giving remixmate write access to data/ and outputs/ only..."
mkdir -p /opt/remixmate/data /opt/remixmate/outputs
chown -R remixmate:remixmate /opt/remixmate/data /opt/remixmate/outputs

echo "==> Installing systemd service..."
cp "$SCRIPT_DIR/remixmate.service" /etc/systemd/system/
echo "    Copied remixmate.service"
systemctl daemon-reload

if [ ! -f /etc/remixmate.env ]; then
  cat <<EOF

/etc/remixmate.env not found; remixmate NOT started.

1. Copy and fill environment file:
     cp $SCRIPT_DIR/remixmate.env.example /etc/remixmate.env
     nano /etc/remixmate.env
     chmod 600 /etc/remixmate.env

2. Replace YOUR_DOMAIN in $SCRIPT_DIR/Caddyfile

Then re-run this script.
EOF
  exit 1
fi

echo "==> Installing Caddyfile..."
if grep -q YOUR_DOMAIN "$SCRIPT_DIR/Caddyfile"; then
  echo "ERROR: replace YOUR_DOMAIN in $SCRIPT_DIR/Caddyfile, then re-run this script" >&2
  exit 1
fi
systemctl enable --now caddy
install -m 644 "$SCRIPT_DIR/Caddyfile" /etc/caddy/Caddyfile
systemctl reload caddy
echo "    Caddy reloaded"

echo "==> Restarting remixmate..."
systemctl enable remixmate
systemctl restart remixmate

echo "==> Readiness (/health/ready):"
if ! curl -sS --retry 10 --retry-connrefused --retry-delay 2 -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/health/ready; then
  echo "⚠️  /health/ready did not answer; check: journalctl -u remixmate -n 50"
fi

echo ""
echo "Bootstrap complete!"
