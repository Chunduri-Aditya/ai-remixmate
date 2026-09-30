# AGENTS.md

For Codex and any other non Claude agent. Rules live in the files below, never here.

1. Read `CLAUDE.md` (what the repo is, which doc to trust), then `RULES.md` (commands, do and do not), then the map for your area: `docs/agents/UI_MAP.md`, `docs/agents/MIXING_MAP.md` or `ARCHITECTURE.md`.
2. Known problems are in `docs/agents/BACKLOG.md`; cite an id instead of filing the same problem again.

What differs for you:
- No Claude Code hooks run for you: nothing loads `MEMORY.md` at session start and nothing checks your commits.
- The permission deny rules in `.claude/` do not apply to you. Nothing blocks a destructive route call or a delete on your behalf, so the Do not list in `RULES.md` is yours to keep.
- `MEMORY.md`, `graphify-out`, `.claude/` and the data dirs may be absent in your checkout; read the code instead.
