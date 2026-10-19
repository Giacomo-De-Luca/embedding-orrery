# Categorical Palettes

The categorical palette registry lives in `embedding_visualization/lib/utils/categoryPalettes.ts`
(`CATEGORY_PALETTES`). Adding an entry is all that's needed: `ColorScaleSelector` iterates
`BUILTIN_PALETTE_NAMES` for the picker (showing the first eight swatches and the colour count),
`generateCategoryColors` cycles the array when the category count exceeds the palette length, and
the chosen key round-trips through the URL and per-collection default colour schemes as a plain
string. `DEFAULT_PALETTE_KEY` stays `cosmicGalaxy`.

## Slot order matters

Categories are sorted (`CategoryKey.sort`) and take palette slots in order, so a field with four
values is coloured by slots 0–3 whatever the palette holds further on. Every palette is therefore
ordered **prefix-first**: short prefixes are as mutually distinct as the design allows. In a scatter
plot any two categories can sit side by side, so the metric is the minimum *all-pairs* separation
of the first *k* colours, not the separation of neighbouring slots (which only matters in the legend).

Separation is measured in OKLab (Euclidean distance ×100; about 2 is a just-noticeable difference,
≥15 is comfortably distinct) and under Machado–Oliveira–Fernandes protan/deutan simulation.
`ColorMath` (`lib/utils/colorMath.ts`) implements the maths; `PaletteAudit`
(`lib/utils/paletteDesign.ts`) prints the numbers below.

## Palettes (2026-10)

| Key | Label | Colors | Intent |
|---|---|---|---|
| `cosmicGalaxy` | Cosmic Galaxy | 20 | Default. Ten saturated mains first, their pale echoes after (slot *i*+10 echoes slot *i*). Tuned for the black starfield; echoes drop to 1.3–2.4:1 on white, so use the Light variant in light mode. |
| `cosmicGalaxyLight` | Cosmic Galaxy Light | 20 | Same families in the same order; the echo slots hold deep siblings so **every slot is ≥3.35:1 on white**. |
| `cosmicGalaxyXL` | Cosmic Galaxy XL | 32 | First 20 identical to `cosmicGalaxy`, then six more mains (sea-green, lime, grape, copper, wine, citron) and their echoes. |
| `deepField` | Deep Field | 64 | **For maps with more than 20 topics.** Saturated colours over the whole hue circle, picked farthest-first; opens with the default's teal and gold. |
| `aurora` | Aurora | 16 | Polar lights: oxygen greens and limes, teal, violet and magenta curtains. No reds or oranges. |
| `hubble` | Hubble Palette | 16 | SHO false colour: sulphur golds, ambers and rusts against oxygen teals, cyans and sapphire blues. |
| `pulsar` | Pulsar | 14 | Electric colours at the sRGB chroma ceiling, bright enough to glow on black. |
| `emeraldGalaxy` / `azureGalaxy` / `emberGalaxy` / `violetGalaxy` | … Galaxy | 12 | Thematic single-family sweeps. Pretty, but by construction their all-pairs minima are tiny (see audit); use them for ≤4 categories. |
| `Galaxy` | Galaxy | 20 | Legacy variant of the original interleaved Cosmic Galaxy with a grey leading pair; kept for collections whose saved default names it. |
| `category10` / `category20` | D3 | 10/20 | Stock D3 palettes. |

### Cosmic Galaxy family table

The three Cosmic palettes are derived from one table, `COSMIC_FAMILIES` (`main`, `echo`,
`lightMain`, `lightDeep` per hue family), and one slot order, `COSMIC_FAMILY_ORDER`:

```
teal · gold · red · violet · coral · green · blue · indigo · magenta · amber
```

Before 2026-10 the default interleaved main/echo pairs (`teal, teal-echo, gold, gold-echo, …`), so a
two-category field got teal and a paler teal. The order is now **all mains first, echoes after**, and
the mains follow the farthest-first order from teal computed by `PaletteDesigner.prefixOrder` (a
test pins it). Two, three or four categories now receive teal/gold/red/violet (all-pairs ΔE 22.0,
CVD 14.1) instead of teal and its echo (ΔE 31 but CVD-adjacent blues from slot 3 on). Saved colour
schemes store only the palette key, so existing collections pick up the new order automatically.

### Generated palettes

`deepField`, `aurora`, `hubble` and `pulsar` are **computed, not hand-picked**. Each has a recipe in
`lib/utils/paletteRecipes.ts`:

1. `PaletteDesigner.candidates` lays a grid over OKLCH (lightness band, hue step, chroma as a
   fraction of the in-gamut maximum at that point), restricted to the recipe's **hue arcs**, and keeps
   only in-gamut colours with ≥3:1 contrast on black, a floor contrast on white, chroma above the
   "reads as grey" floor, and ΔE ≥12 from the reserved Missing (`#7f7f7f`) and Unclustered (`#e416be`)
   colours.
2. `PaletteDesigner.farthestFirst` starts from the recipe's seed and repeatedly adds the candidate
   whose minimum **separation** from everything chosen is largest. Separation is
   `min(ΔE, 1.6·ΔE_protan, 1.6·ΔE_deutan)`, so a pair that collapses for red–green deficient readers
   counts as close. Greedy farthest-first is what makes every prefix well spread.

Regenerate with `npm run palettes:design` (all recipes, or `-- aurora` for one) and paste the printed
arrays into the registry. `lib/utils/__tests__/paletteDesign.test.ts` regenerates each recipe and
fails if the committed colours drift. Hand-tune a generated palette by changing its recipe (arcs,
seed, lightness) rather than its hexes.

## Audit (all-pairs minima of the first *k* colours: plain ΔE / worst protan-or-deutan ΔE)

From `npm run palettes:design`, 2026-10:

| Palette | n | min contrast black / white | k=4 | k=8 | k=12 | k=16 | k=20 | k=32 | k=64 |
|---|---|---|---|---|---|---|---|---|---|
| cosmicGalaxy | 20 | 3.10 / 1.34 | 22.0 / 14.1 | 8.7 / 5.7 | 6.1 / 3.8 | 2.7 / 2.6 | 2.7 / 0.6 | | |
| cosmicGalaxyLight | 20 | 2.62 / 3.35 | 17.5 / 5.2 | 6.6 / 0.9 | 6.6 / 0.9 | 5.2 / 0.9 | 4.0 / 0.4 | | |
| cosmicGalaxyXL | 32 | 3.10 / 1.34 | 22.0 / 14.1 | 8.7 / 5.7 | 6.1 / 3.8 | 2.7 / 2.6 | 2.7 / 0.6 | 2.7 / 0.4 | |
| deepField | 64 | 3.13 / 1.61 | 28.7 / 18.2 | 15.0 / 9.4 | 12.9 / 8.3 | 11.0 / 7.0 | 9.4 / 5.5 | 6.9 / 4.0 | 4.5 / 2.7 |
| aurora | 16 | 3.65 / 1.51 | 24.7 / 18.1 | 14.0 / 8.8 | 11.2 / 8.2 | 8.4 / 5.4 | | | |
| hubble | 16 | 3.25 / 1.50 | 29.1 / 25.2 | 11.7 / 9.2 | 10.8 / 7.4 | 7.6 / 5.1 | | | |
| pulsar | 14 | 5.07 / 1.34 | 27.6 / 19.4 | 16.0 / 9.3 | 11.0 / 6.3 | | | | |
| emeraldGalaxy | 12 | 3.34 / 1.77 | 4.6 / 4.5 | 3.3 / 2.1 | 2.1 / 2.1 | | | | |
| azureGalaxy | 12 | 3.26 / 1.83 | 6.0 / 5.0 | 3.4 / 3.4 | 2.2 / 1.6 | | | | |
| emberGalaxy | 12 | 3.07 / 1.88 | 4.9 / 4.3 | 4.4 / 2.1 | 2.9 / 1.1 | | | | |
| violetGalaxy | 12 | 3.14 / 1.96 | 8.5 / 3.8 | 4.7 / 0.9 | 2.6 / 0.9 | | | | |
| Galaxy | 20 | 3.10 / 1.16 | 11.9 / 11.7 | 11.8 / 6.4 | 11.1 / 3.6 | 5.9 / 2.1 | 5.5 / 0.6 | | |
| category20 | 20 | 3.55 / 1.45 | 13.5 / 11.4 | 10.4 / 0.7 | 6.2 / 0.7 | 5.5 / 0.7 | 5.4 / 0.7 | | |

Reading the table: at 20 categories Deep Field keeps every pair ≥9.4 apart where Cosmic Galaxy has
two pairs at 2.7 (below the just-noticeable threshold for some viewers); at 32 it is 6.9 against
XL's 2.7. For 20+ categories prefer Deep Field.

### Known weak pairs (pre-existing, kept for identity with the default)

- Cosmic mains **coral `#e8590c` and amber `#e67700`**: ΔE 6.1 (CVD 3.8). Amber is deliberately the
  last main (slot 9) so only fields with ≥10 categories meet both.
- Cosmic echoes **salmon `#ffa094` and rose `#ffa8a8`**: ΔE 2.7; **lavender `#b197fc` and
  silver-blue `#91a7ff`**: CVD 0.6. These sit in slots 10–19 and are why the audit collapses from
  k=16. Replacing the echo tier is the natural next step if the default must carry 20 categories.
- `#0b7285` (deep nebula teal) sits below the OKLCH 0.10 chroma floor at C 0.087. This is a gamut
  bound, not a defect: the sRGB ceiling for deep teal is ~0.085–0.10 across L 0.50–0.56. It is kept
  verbatim for identity with the default palette (and seeds Deep Field).

## Validation method

The dataviz-skill validator (`validate_palette.js`: lightness band, OKLCH chroma floor ≥0.10,
adjacent/all-pairs CVD ΔE, WCAG contrast vs surface) was run on every palette on `#000000`
(dark-first palettes) or `#ffffff` (`cosmicGalaxyLight`). Its pass thresholds (all-pairs ΔE ≥15,
CVD ≥8) are calibrated for ≤8 bar/line series and are unreachable for 14–64 colours in sRGB, so for
these scatter palettes the contract is instead: **contrast ≥3:1 on the design surface for every
slot**, **chroma ≥0.10** (except the documented teal), and the **max-min packing** reported above.
Contrast passes for every palette; chroma passes for every colour except `#0b7285` and the amber echo
`#ffd8a8` (C 0.076, inherited).

Cluster-label titles are drawn in the raw category colour with only mild desaturation
(`desaturateHex(color, 0.35)` toward `#606060` in light mode — `ScatterPlot3D.tsx`), so palette
contrast against the surface is what makes titles legible; that's why the light palette enforces
the ≥3:1 floor on every slot, not just the "main" slots.
