---
name: stem-processing
description: Demucs stem separation and library hygiene for AI RemixMate. Use when separating tracks into stems, tuning the separation config, debugging missing/failed stems, pruning unusable library songs, or wiring stems into the remix bridge. Project-scoped to ai-remixmate.
---

# Stem Processing (Demucs)

Separate tracks into stems reliably; keep the library clean. CPU is slow — plan for it.

## Where things live
- **Config:** `config.yaml` → `separation:` block
  - `model: htdemucs` (also `htdemucs_ft`, `mdx_extra`)
  - `device: auto` (`cpu` | `cuda` | `mps` for Apple Silicon)
  - `keep_raw_after_separation`, `prune_on_download`, `max_vram_gb`
- **Download + optional separation:** `scripts/download.py` (skips split if `demucs` not installed)
- **Bridge integration:** `scripts/remixmate_cli.py` → `generate_stems_and_params(base, match, out_dir)`
- **Cleanup:** `scripts/purge_library.py` (flags songs with no full mix and no stems)

## Rules
- **Set the device deliberately.** On Apple Silicon use `device: mps`. Demucs on CPU runs 2–5× real time — batch and expect latency.
- **Respect the VRAM budget** (`max_vram_gb`, default 12) so separation and other AI models don't OOM.
- **Missing stems ≠ crash.** A song folder with no `full.wav` and no stems is a failed/shell download — route it to `purge_library.py`, don't silently produce empty remixes.
- **Deterministic paths, code-owned.** Stem paths and mix params come from the CLI helper, not model guesses.
- **Never commit keys** — `config.local.yaml` is gitignored; keep secrets there.

## Verify
```bash
python -m scripts.api.main                       # backend
pytest tests/ -q -m "not dj_analysis"            # 226 tests
python scripts/purge_library.py                  # list unusable songs (review before deleting)
```

## Checklist
- [ ] Device matches the machine (mps on Apple Silicon)?
- [ ] VRAM budget respected?
- [ ] Missing-stem songs handled, not silently mixed?
- [ ] Stem paths from the CLI helper, not hardcoded guesses?
