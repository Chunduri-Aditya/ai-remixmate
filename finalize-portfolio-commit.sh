#!/usr/bin/env bash
# ============================================================
#  AI RemixMate — finalize portfolio commit
#
#  Run from repo root:  bash finalize-portfolio-commit.sh
#
#  This is a one-time script. It removes itself at the end.
# ============================================================
set -e

cd "$(dirname "$0")"

echo "▶ 1. Removing the previous one-time cleanup script (it shouldn't be in the repo)..."
if git ls-files --error-unmatch cleanup-and-commit.sh >/dev/null 2>&1; then
  git rm -f cleanup-and-commit.sh
elif [ -f cleanup-and-commit.sh ]; then
  rm -f cleanup-and-commit.sh
fi

echo "▶ 2. Sanity-checking the README..."
LINES=$(wc -l < README.md | tr -d ' ')
echo "   README.md is now $LINES lines (was 586)"

echo "▶ 3. Staging changes..."
git add README.md docs/README.md
git add -A  # picks up the cleanup-and-commit.sh removal

echo "▶ 4. Showing what's about to be committed..."
git status --short

echo
echo "▶ 5. Diff stat:"
git diff --cached --stat

echo
read -p "▶ 6. Proceed with commit? [y/N] " yn
case $yn in
  [Yy]* )
    git commit -m "docs: rewrite README for portfolio + slim docs index

- Rewrite README.md from scratch:
  * Cut from 586 lines to ~225
  * Lead with concrete technical claims, not marketing copy
  * Add 'parts that are actually interesting' section explaining
    the four hard problems (beat-grid lock, stem-aware crossfade,
    dynamic EQ fade, procedural bridge beats)
  * Add 'why I built this' section
  * Add tech-stack table, status section, honest caveats
  * Strip AI-detector vocabulary

- Add docs/README.md as a clean index for the docs/ folder

- Remove cleanup-and-commit.sh (one-time tooling, doesn't
  belong in the repo)"
    echo "   ✓ Commit created"
    ;;
  * )
    echo "   ✗ Aborted before commit. Working tree is staged."
    exit 0
    ;;
esac

echo
echo "▶ 7. Recent history:"
git log --oneline -5

echo
echo "═══════════════════════════════════════════════════════════"
echo " NEXT: push to origin"
echo "═══════════════════════════════════════════════════════════"
echo
echo "   git push origin main"
echo
echo " Then check it on GitHub:"
echo "   https://github.com/Chunduri-Aditya/ai-remixmate"
echo
echo " Optional: also fix the commit author identity so the"
echo " contribution graph fills in properly:"
echo
echo "   git config --global user.name \"Aditya Chunduri\""
echo "   git config --global user.email \"aditya2210.a1@gmail.com\""
echo
echo "═══════════════════════════════════════════════════════════"

# Self-cleanup: this script shouldn't live in the repo either.
# Removing from disk now. Not staged — git won't track it because
# it never existed in the index.
echo
echo "▶ 8. Removing this script from disk (it was one-time tooling)..."
rm -- "$0"
echo "   ✓ Done."
