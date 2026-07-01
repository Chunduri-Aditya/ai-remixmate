# Testing Plan

Testing is split into deterministic TypeScript and Python suites. TypeScript tests cover audio primitives and planning logic with Vitest. Python tests cover pure compatibility, transition planning, recipes, automix, and track ranking with pytest. Optional audio analysis functions are intentionally not required for base tests. Risks include unavailable local JS dependencies and pytest cache writes; commands should run inside the lab folders. Future CI should add schema validation and typechecking after isolated package dependencies are installed.

## Implementation Strategy

Keep the lab code independent and deterministic. Use schemas and tests as the contract before adding adapters.

## Risks

The highest risk is treating prototype logic as production-ready without validating it against real audio and current app state.

## Validation

Run the TypeScript and Python tests from inside this folder, then perform Git no-touch verification from the repository root.

## Future Integration Steps

Port one small module at a time, preserve existing behavior, and add compatibility shims rather than replacing production surfaces in one change.
