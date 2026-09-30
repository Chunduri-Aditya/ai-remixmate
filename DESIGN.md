# DESIGN

_How the web UI in `frontend/` looks today, so generated code matches it. Committed. This records the implemented style, not a chosen direction. Colour values are deliberately absent: read them in `frontend/src/styles/tokens.css`, the single source._

## Direction
- No direction has been chosen through a design pass. The current look is what the code implements: dark surfaces with amber, ice, green, crimson and violet accents (the colour groups in `frontend/src/styles/tokens.css`, each commented with its role).
- A redesign starts by proposing directions and picking one before any markup changes. Until then, new UI follows the rules below.

## Theme
- Dark only. There is no light theme, no theme switch, and no `prefers-color-scheme` rule anywhere in the frontend CSS. The page declares no `color-scheme` either: UI-36.
- `frontend/src/styles/global.css` honours `prefers-reduced-motion`.

## Tokens
- `frontend/src/styles/tokens.css` holds every primitive as a `:root` custom property: colour, font families, type scale, spacing, radius, shell and player bar sizes, shadows, easing and durations, and the z index scale. Consume them as `var(--token-name)`; check there before writing any colour or spacing value.
- `frontend/src/styles/global.css` imports the tokens and holds the reset and shared utilities; `frontend/src/main.tsx` imports `global.css`.
- The palette summary and shell layout are in `DEVELOPING.md`, "Design system". It quotes values, so when it and the tokens file disagree, the tokens file wins.

## Typography
- Three families, each a token: a display face, a UI face and a mono face (Space Grotesk, Inter, JetBrains Mono). They load from Google Fonts in `frontend/index.html`; the token stacks fall back to system fonts.

## Styling rules
- Plain CSS, one file per page or component, imported by its TSX file. Exceptions: `frontend/src/components/WaveformDeck.tsx` has no CSS file and takes its `md-waveform*` classes from `frontend/src/pages/MixDeck.css` (unstyled on any other page), `CamelotWheel` is inline SVG only, and `RootErrorBoundary` reuses `PageErrorBoundary.css`. No Tailwind and no CSS modules, so class names are global and a per file prefix is the only namespace (`mc-` for Mission Control, `md-` for Mix Deck, and so on). The prefix table is in `docs/agents/UI_MAP.md`.
- New pages follow the `frontend/src/pages/PageBase.css` frame: `page-base`, then `page-base__header`, then `page-base__body`. The full recipe (route, rail item, nav type) is in `DEVELOPING.md`, "Common tasks".

## Known hardcoded colours
Literal copies of token values that do not follow a token change. Canvas and WaveSurfer cannot read CSS variables, so the script copies must be edited by hand; no backlog id tracks them.

- Script: `frontend/src/pages/MixDeck.tsx::DECK_HEX`, the WaveSurfer colours in `frontend/src/pages/MixVault.tsx::WaveformPlayer`, canvas fills in `frontend/src/components/PlayerBar.tsx::MiniViz`, `hsl()` colours in `frontend/src/components/CamelotWheel.tsx::CamelotWheel` and `frontend/src/pages/Widget.tsx::camelotHue`, the white label fills (`#fff`, `rgba(255,255,255,0.75)`, `rgba(255,255,255,0.3)`) in `CamelotWheel`, and the `rgba(0,0,0,0.2)` badge background in `frontend/src/pages/MixVault.tsx::MixCard`.
- CSS with literal hex outside the tokens file: `frontend/src/shell/LeftRail.css`, `frontend/src/components/PageErrorBoundary.css`, `frontend/src/components/PlayerBar.css`, `frontend/src/pages/LibraryAtlas.css`, `frontend/src/pages/LegalPage.css`.
- Markup: the `theme-color` meta in `frontend/index.html`.
