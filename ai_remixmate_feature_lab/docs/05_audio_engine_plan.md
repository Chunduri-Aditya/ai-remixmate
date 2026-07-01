# Audio Engine Plan

The lab implements audio math primitives only. Live playback, Web Audio nodes, and Python DSP renderers remain future integration work. Primitives cover gain, crossfader curves, EQ/filter validation, tempo math, beat sync, quantize, loop math, phrase math, and automation interpolation. Validation should happen before any renderer receives values. Risks include clipping, unsafe tempo shifts, wrong beatgrid phase, and discontinuous automation. Validation uses deterministic unit tests for numeric ranges and edge cases. Future integration should adapt these functions into the existing renderer and React controls behind adapters.

## Implementation Strategy

Keep the lab code independent and deterministic. Use schemas and tests as the contract before adding adapters.

## Risks

The highest risk is treating prototype logic as production-ready without validating it against real audio and current app state.

## Validation

Run the TypeScript and Python tests from inside this folder, then perform Git no-touch verification from the repository root.

## Future Integration Steps

Port one small module at a time, preserve existing behavior, and add compatibility shims rather than replacing production surfaces in one change.
