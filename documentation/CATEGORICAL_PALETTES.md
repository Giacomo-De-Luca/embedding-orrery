# Categorical Palettes

The categorical palette registry lives in `embedding_visualization/lib/utils/categoryPalettes.ts`
(`CATEGORY_PALETTES`). Adding an entry is all that's needed: `ColorScaleSelector` iterates
`BUILTIN_PALETTE_NAMES` for the picker, `generateCategoryColors` cycles the array when the
category count exceeds the palette length, and the chosen key round-trips through the URL and
per-collection default color schemes as a plain string. `DEFAULT_PALETTE_KEY` stays `cosmicGalaxy`.

## Palettes (2026-07)

| Key | Label | Colors | Intent |
|---|---|---|---|
| `cosmicGalaxy` | Cosmic Galaxy | 20 | Default. Ten hue families as main + pale-echo pairs, tuned for the black starfield. Pale echoes drop to 1.3–2.4:1 on white — use the Light variant in light mode. |
| `cosmicGalaxyLight` | Cosmic Galaxy Light | 20 | Same ten families in the same order; echoes replaced by deep siblings so **every slot holds ≥3.3:1 on white** — points and cluster-label titles stay readable in light mode. |
| `cosmicGalaxyXL` | Cosmic Galaxy XL | 32 | First 20 identical to `cosmicGalaxy` (switching keeps existing colors), then six more main+echo families (sea-green, lime, grape, copper, wine, citron). Cycling starts at 33 categories instead of 21. |
| `emeraldGalaxy` | Emerald Galaxy | 12 | Thematic single-family green. |
| `azureGalaxy` | Azure Galaxy | 12 | Thematic blues, cobalt → ice. |
| `emberGalaxy` | Ember Galaxy | 12 | Thematic warm sweep, garnet → gold. |
| `violetGalaxy` | Violet Galaxy | 12 | Thematic violets/orchids. |
| `Galaxy` | Galaxy | 20 | Legacy variant of cosmicGalaxy with gray leading pair. |
| `category10` / `category20` | D3 | 10/20 | Stock D3 palettes. |

Thematic palettes carry category separation inside one hue family via strong dark/bright
lightness alternation plus sub-hue shifts; they are tuned dark-first (all ≥3:1 on black) but
stay mid-range enough to remain visible on white.

## Validation method

New palettes were designed in OKLCH and checked computationally (never by eye) with the
dataviz-skill validator (lightness band, OKLCH chroma floor ≥0.10, adjacent-pair CVD ΔE under
Machado protan/deutan/tritan simulation, WCAG contrast vs. surface). Surfaces: `#ffffff`
(light plots) and `#000000` (dark starfield). Results:

- `cosmicGalaxyLight`: all 20 slots ≥3.35:1 on white; worst adjacent CVD ΔE 12.6 (target ≥12); lightness band pass.
- `cosmicGalaxyXL`: all 32 ≥3:1 on black; worst adjacent CVD ΔE 22.5. Bright echoes intentionally exceed the dark lightness band (starfield aesthetic, inherited from `cosmicGalaxy`).
- Thematic ×4: all ≥3.07:1 on black; worst adjacent CVD ΔE 14.3–19.4.

**Known exception**: `#0b7285` (deep nebula teal, anchor of both cosmic palettes) sits below
the 0.10 OKLCH chroma floor at C 0.087. This is a gamut bound, not a defect: the sRGB ceiling
for deep teal is ~0.085–0.10 across L 0.50–0.56, so no more-saturated deep teal exists. It is
kept verbatim for identity with the default palette.

Cluster-label titles are drawn in the raw category color with only mild desaturation
(`desaturateHex(color, 0.35)` toward `#606060` in light mode — `ScatterPlot3D.tsx`), so palette
contrast against the surface is what makes titles legible; that's why the light palette enforces
the ≥3:1 floor on every slot, not just the "main" slots.
