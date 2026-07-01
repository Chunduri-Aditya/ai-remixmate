# Read-Only Integration Plan

Integration must begin by reading current production files and comparing contracts. Safe integration points are FastAPI schemas and routers for new analysis endpoints, task modules for async work, core modules for compatibility and planning, frontend API wrappers, shared types, MixDeck, SignalSearch, SetBuilder, and MixVault surfaces. Do not patch those files from this lab. Instead, create a future integration branch, port tests first, add adapters, run root tests, and only then wire UI. This folder remains the source reference until integration is reviewed.

## Implementation Strategy

Keep the lab code independent and deterministic. Use schemas and tests as the contract before adding adapters.

## Risks

The highest risk is treating prototype logic as production-ready without validating it against real audio and current app state.

## Validation

Run the TypeScript and Python tests from inside this folder, then perform Git no-touch verification from the repository root.

## Future Integration Steps

Port one small module at a time, preserve existing behavior, and add compatibility shims rather than replacing production surfaces in one change.
