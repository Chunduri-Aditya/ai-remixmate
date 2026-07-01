# Frontend Component Plan

React prototypes are standalone typed components with inline styles and no dependency on the existing app. They demonstrate deck state, waveform visualization, mixer controls, cues, loops, compatibility, transition plans, recipes, stems, and automix queues. They are not production UI. Risks include visual mismatch with the current design system and duplicated state logic. Future integration should restyle them with existing tokens, connect to Zustand/API adapters, and add interaction tests before replacing any existing page sections.

## Implementation Strategy

Keep the lab code independent and deterministic. Use schemas and tests as the contract before adding adapters.

## Risks

The highest risk is treating prototype logic as production-ready without validating it against real audio and current app state.

## Validation

Run the TypeScript and Python tests from inside this folder, then perform Git no-touch verification from the repository root.

## Future Integration Steps

Port one small module at a time, preserve existing behavior, and add compatibility shims rather than replacing production surfaces in one change.
