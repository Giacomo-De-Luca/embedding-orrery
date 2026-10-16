# Explore header layout (mobile, tablet, desktop, HF Space)

How the Explore page's header (`AppHeader`) arranges itself per viewport and
build, and how the floating chrome below it (panels, legend, modebar, corner
chips) stays clear of it.

## Why it changed (2026-10-16)

The old header was a flex row forced onto one line from `sm` (640 px), with a
second right-aligned row in demo builds and a fixed spacer for the Hugging Face
pill. Screenshots of the live Space and the local build showed:

- **640–1279 px** (tablets, landscape phones, small laptops): the semantic
  search was squeezed to ~70 px and hidden under the collection selector; nav
  pills spilled off-screen in the full build; the demo's second row collided
  with an open panel.
- **Phones in the Space**: the 160 px pill spacer truncated the collection
  name for nothing (HF shows no top pill below 640 px), and HF's corner button
  covered the collapsed search-results chip.
- **Every width ≥ 640 px in the Space**: the 288 px spacer was ~80 px short of
  the 368 px pill, which covered the collection selector's chevron, desktop
  included.
- **Full build on phones**: the modebar (pinned at `top: 80px`) sat on the
  nav pills.

Panel, legend and modebar tops were separate hard-coded numbers
(`top-34 sm:top-20`, `top-30`/`top-40`, `80px`/`120px`) that had to be kept in
step with the header by hand.

## Hugging Face chrome (measured 2026-10-16)

On `huggingface.co/spaces/GiacomoDeLuca/orrery-demo` the app runs in a full-page
iframe and HF floats its own chrome over it:

| Viewport | HF chrome over the app |
|---|---|
| < 640 px | 38 × 38 px button, 16 px from the bottom-right corner |
| ≥ 640 px | 368 × 40 px "owner / space, likes" pill at top 20–60 px, 24 px from the right edge |

The pill's width depends on the Space name and like count, so the clearance
(`--hf-pill-clearance: 25rem`) keeps ~30 px of slack.

## Layout

One CSS grid with four cells, keyed by `data-header-cell`: **toggles** (panel
buttons, plus the point/dimension badges from `lg`), **search**,
**collection**, **actions** (demo `?` button; `HeaderMenu` below `xl`; GitHub,
theme and `PageNav` from `xl`). Placement lives only in `app/globals.css`
(`.explore-header`). The TSX just renders the cells.

| Width | Full build | Demo build |
|---|---|---|
| < 640 | `toggles search` / `actions collection`: an icon column (burger first, under the first toggle) beside an input column, so search and collection share both edges | same |
| 640–1279 | one row: `toggles search collection actions` | `toggles` (beside the HF pill) / `search collection actions` |
| ≥ 1280 | one row, actions show GitHub + theme + nav | `toggles search collection` (clear of the pill) / `actions` right-aligned |

The ≥ 1280 demo layout is the earlier desktop design, now with the correct pill
clearance.

The `?` button stays visible at every width because it is the demo's way back
to the mission menu. Its tour anchor (`intro-button`) is therefore always
mounted. `HeaderMenu` lists the pages from `buildPageNavItems` (shared with
`PageNav`). The demo's disabled Collections entry spells out its reason, since
touch has no hover tooltip.

## Offsets for the floating chrome

Grid rows have fixed heights, so the header's height per breakpoint is known
statically. `.explore-shell` (on the page's `<SidebarInset>`) publishes it as
CSS variables:

| Variable | Consumer | < 640 | Full ≥ 640 | Demo 640–1279 | Demo ≥ 1280 |
|---|---|---|---|---|---|
| `--explore-header-h` | (derivations) | 88 | 72 | 116 | 112 |
| `--explore-panel-top` | `explorePanelClass` | 96 | 80 | 124 | 80 (row 2 is empty on the left) |
| `--explore-modebar-top` | `.modebar-container` | 96 | 80 | 124 | 120 |
| `--explore-legend-top` | legend + collapsed pill | 136 | 120 | 164 | 160 |
| `--explore-corner-inset` | collapsed results chip, results table's bottom padding | 16 (64 embedded) | 16 | 16 | 16 |

Desktop values (full ≥ 640, demo ≥ 1280) are unchanged from before. Pages
other than Explore don't set the variables, so `.modebar-container` falls back
to 80 px there.

Shell attributes:

- `data-demo`: from `IS_DEMO`, a build-time constant, so SSR and hydration
  agree.
- `data-hf-embedded`: from `useInHfSpace()`
  (`lib/hooks/useInHfSpace.ts`). This uses `useSyncExternalStore` with a
  `false` server snapshot, so it is hydration-safe.

The header is `pointer-events-none` and only its cells capture input, so the
gaps between controls pass drags through to the plot.

## Files

- `app/globals.css`: the "Explore page chrome" block (grid templates,
  variables, modebar rule)
- `app/components/AppHeader.tsx`: the four cells
- `app/components/HeaderMenu.tsx`: compact menu below `xl`
- `lib/utils/pageNavItems.ts`: page list shared by `PageNav` and `HeaderMenu`
  (unit-tested)
- `lib/hooks/useInHfSpace.ts`, `lib/hooks/useThemeToggle.ts`
- `lib/utils/explorePanel.ts`, `app/components/DashboardPanel.tsx`: the
  consumers
- `app/page.tsx`: sets `explore-shell` and its data attributes

## Verification

The layout is CSS keyed on the shell attributes, so a non-demo dev server can
exercise all three variants (full / demo / demo embedded) by flipping
`data-demo` / `data-hf-embedded` in the DOM. A 36 px placeholder is inserted
in the actions cell to stand in for the demo-only `?` button. A Playwright script did this at
360, 390, 640, 700, 768, 1024, 1280 and 1440 px and asserted:

- the four cells stay on-screen and don't overlap;
- the search is at least 150 px wide and the selector at least 150 px;
- no header control intersects the measured HF pill rect when embedded;
- the panels and the modebar start below any header control above them.
- on phones, the search and collection inputs share both edges and the menu
  button sits under the first panel toggle.

All 24 cases passed. Before the change, the full build failed 6 of its 8
widths. A real demo build (`NEXT_PUBLIC_DEMO_MODE=1`) has not been run against
it. Below 360 px (e.g. 320 px phones) the search input drops to ~140 px but
nothing overlaps.

## Known gaps

- On phones an open panel (full width there) covers the modebar until it is closed.
- `useZoomLimit` is wheel-only, so pinch zoom is unbounded.
- The SAE context menu needs a right-click.
- 3D + haze memory on mobile Safari is unmeasured.
