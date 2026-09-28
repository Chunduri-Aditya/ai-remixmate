#!/usr/bin/env bash
set -euo pipefail

# RemixMate VPS bootstrap (idempotent)
# Mirrors PROVISION.md §2. Run as root on Debian/Ubuntu.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing system packages..."
apt update
apt install -y ffmpeg libsndfile1 caddy postgresql

# Try python3.12-venv, fall back to python3-venv
if apt-cache show python3.12-venv &>/dev/null; then
  apt install -y python3.12-venv
else
  echo "    python3.12-venv not found, falling back to python3-venv"
  apt install -y python3-venv
fi

echo "==> Creating system user remixmate..."
if ! id remixmate &>/dev/null; then
  useradd -r -m -d /opt/remixmate remixmate
  echo "    Created user remixmate"
else
  echo "    User remixmate already exists"
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

echo "==> Setting up repository..."
cd /opt/remixmate
chown -R remixmate:remixmate /opt/remixmate

sudo -u remixmate bash <<'EOF_USER'
  set -euo pipefail
  cd /opt/remixmate

  echo "    Checking out hosted-app branch..."
  git checkout hosted-app

  echo "    Creating virtual environment..."
  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi

  echo "    Installing Python dependencies..."
  .venv/bin/pip install --quiet -r requirements.txt
  .venv/bin/pip install --quiet -e .
  echo "    (Skipped [generative] and [wordplay] extras)"
EOF_USER

echo "==> Installing systemd service..."
cp "$SCRIPT_DIR/remixmate.service" /etc/systemd/system/
echo "    Copied remixmate.service"

cat <<EOF

==> Next steps:

1. Copy and fill environment file:
     cp $SCRIPT_DIR/remixmate.env.example /etc/remixmate.env
     nano /etc/remixmate.env
     chmod 600 /etc/remixmate.env

2. Configure Caddy:
     nano $SCRIPT_DIR/Caddyfile  # Replace YOUR_DOMAIN
     cp $SCRIPT_DIR/Caddyfile /etc/caddy/Caddyfile

3. Reload and enable services:
     systemctl daemon-reload
     systemctl enable remixmate caddy

EOF

if [ ! -f /etc/remixmate.env ]; then
  echo "⚠️  /etc/remixmate.env not found — services NOT started"
  echo "   Complete steps 1-2 above, then: systemctl start remixmate caddy"
else
  systemctl daemon-reload
  systemctl enable remixmate caddy
  echo "✓  Services enabled (already running or ready to start)"
fi

echo ""
echo "Bootstrap complete!"
