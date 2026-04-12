#!/usr/bin/env bash
# generate.sh — Create locally-trusted TLS certificates for AI RemixMate.
#
# Two methods (picks the best one automatically):
#   1. mkcert  — creates certs trusted by your OS/browser (no warnings)
#   2. openssl — self-signed fallback (browser shows a one-time warning)
#
# Usage:
#   cd certs/ && bash generate.sh
#
# Output:
#   cert.pem  — TLS certificate
#   key.pem   — Private key

set -e

CERT_DIR="$(cd "$(dirname "$0")" && pwd)"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

# Detect LAN IP for SAN (Subject Alternative Name)
if command -v ipconfig &>/dev/null; then
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "")
else
  LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")
fi
[ -z "$LAN_IP" ] && LAN_IP="127.0.0.1"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   AI RemixMate — TLS Certificate Generator   ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  LAN IP detected: $LAN_IP"
echo ""

# ── Method 1: mkcert (preferred — zero browser warnings) ──────────────────

if command -v mkcert &>/dev/null; then
  echo "✅  Found mkcert — generating locally-trusted certificates..."
  echo ""

  # Install root CA into system trust store (one-time)
  mkcert -install 2>/dev/null || true

  # Generate cert for localhost + LAN IP
  mkcert -cert-file "$CERT_FILE" -key-file "$KEY_FILE" \
    localhost 127.0.0.1 ::1 "$LAN_IP" \
    "$(hostname)" "*.local"

  echo ""
  echo "🔒  Certificates generated (mkcert — fully trusted by your browser):"
  echo "    cert: $CERT_FILE"
  echo "    key:  $KEY_FILE"
  echo ""
  echo "    Your browser will show a green lock — no warnings."
  exit 0
fi

# ── Method 2: OpenSSL fallback (self-signed) ──────────────────────────────

echo "⚠  mkcert not found — falling back to OpenSSL self-signed cert."
echo "   (Install mkcert for zero-warning certs: brew install mkcert)"
echo ""

# Generate a self-signed cert valid for 365 days
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -days 365 \
  -subj "/CN=AI RemixMate/O=RemixMate Local/C=US" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$LAN_IP,IP:::1" \
  2>/dev/null

echo "🔒  Self-signed certificates generated:"
echo "    cert: $CERT_FILE"
echo "    key:  $KEY_FILE"
echo ""
echo "    ⚠ Your browser will show a warning on first visit."
echo "    Click 'Advanced' → 'Proceed to localhost' to bypass it."
echo ""
echo "    💡 To eliminate the warning, install mkcert:"
echo "       brew install mkcert     # macOS"
echo "       sudo apt install mkcert # Ubuntu"
echo "       Then re-run this script."
