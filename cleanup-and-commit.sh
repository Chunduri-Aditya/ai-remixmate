#!/usr/bin/env bash
# ============================================================
#  AI RemixMate — cleanup + commit prep
#  Run this from the repo root: bash cleanup-and-commit.sh
# ============================================================
set -e

cd "$(dirname "$0")"

echo "▶ 1. Removing stale git lock files (if any)..."
rm -f .git/index.lock .git/objects/maintenance.lock || true

echo "▶ 2. Removing .DS_Store files from repo (gitignored anyway, just tidying)..."
find . -name '.DS_Store' \
  -not -path './library/*' \
  -not -path './data/*' \
  -not -path './remix-env/*' \
  -delete 2>/dev/null || true

echo "▶ 3. Verifying legacy move..."
if [ -d scripts/legacy ]; then
  echo "   scripts/legacy still exists — moving to archive/"
  mkdir -p archive
  mv scripts/legacy archive/legacy
fi
echo "   ✓ archive/legacy/ exists: $(find archive/legacy -name '*.py' 2>/dev/null | wc -l | tr -d ' ') python files"

echo "▶ 4. Sanity check what git sees..."
echo "   Untracked: $(git status --porcelain | grep -c '^??')"
echo "   Modified : $(git status --porcelain | grep -c '^ M')"
echo "   Deleted  : $(git status --porcelain | grep -c '^ D')"

echo "▶ 5. Staging everything..."
git add -A

echo "▶ 6. Showing staged summary (file count by top-level dir)..."
git diff --cached --name-only | awk -F/ '{print $1}' | sort | uniq -c | sort -rn

echo
read -p "▶ 7. Proceed with commit? [y/N] " yn
case $yn in
  [Yy]* )
    git commit -m "v0.2.0 — full project sync

Brings the repo up to date with everything built since the
original initial commit:

- FastAPI backend (scripts/api/) with routers, jobs, tasks
- Core engine modules (scripts/core/): dj_engine, stems, key
  detection, mastering, codec tokens, generative remix, etc.
- React + Vite frontend (frontend/)
- Streamlit UI (scripts/ui/app.py)
- Docker + docker-compose
- Pre-commit, pyproject.toml, requirements-dev.txt
- Test suite (tests/) with e2e harness
- Documentation (docs/, GUIDE.md, CHANGELOG.md)
- Pipeline scripts (run_pipeline.sh, run_overnight.sh, start.sh)
- Moved old scripts to archive/legacy/"
    echo "   ✓ Commit created"
    ;;
  * )
    echo "   ✗ Aborted before commit. Working tree is staged."
    exit 0
    ;;
esac

echo
echo "▶ 8. Current state:"
git log --oneline -5
echo
echo "═══════════════════════════════════════════════════════════"
echo " NEXT: push to origin"
echo "═══════════════════════════════════════════════════════════"
echo " Your local main is ahead 7 + your new commit, and behind 2"
echo " on origin. Those 2 'behind' commits on GitHub are old"
echo " 'Initial commit' snapshots from Nov 5 and Nov 11 — they"
echo " are NOT part of your real history."
echo
echo " Recommended: force-push to replace remote with your real"
echo " state. This is safe because:"
echo "   • The repo is yours alone (will be private)"
echo "   • The remote commits are stale duplicates, not real work"
echo
echo " To force-push:"
echo "   git push --force-with-lease origin main"
echo
echo " Then make the repo private at:"
echo "   https://github.com/Chunduri-Aditya/ai-remixmate/settings"
echo "   → scroll to 'Danger Zone' → 'Change repository visibility'"
echo "═══════════════════════════════════════════════════════════"
