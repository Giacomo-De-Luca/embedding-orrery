# Nebula haze: density-based renderer (experimental)

Branch `nebula-density-haze`. A second implementation of the 3D nebula haze,
selectable at runtime next to the existing sprite renderer, built to fix two
failure modes of the sprites without losing the look they produce on WordNet
and Lacan. Nothing about the sprite renderer changed.

## Why

The sprite renderer (`lib/utils/hazeRenderer.ts`) fixes each cluster's *total*
light (sprite count × opacity is constant per cluster) and knows nothing about
other clusters. Two consequences, measured with a numeric replica of the sprite
pipeline on the real UMAP-3D projections:

| Failure | Mechanism | Measured |
|---|---|---|
| Large, spread topics get no haze (EMNLP) | Brightness at a point ≈ budget × (sprite radius / cluster radius)² × boost²; the only term that differs between clusters is the ×3 boost below 200 points, so clusters above the threshold are ~10× dimmer | Spearman(own haze, cluster size) = −0.93; 90th/10th percentile brightness ratio 9.9× (WordNet 2.6×, where every cluster is small) |
| Intermixed classes blow out to white (trec fine labels, 46 classes) | Every class's extent ≈ the whole dataset, so sprites are dataset-sized; K classes stack K full budgets on the same pixels; hues add to achromatic | 21% of lit pixels clipped, 25% near-white |

## How the density renderer works

`lib/utils/densityHaze/` (`README.md` there lists the files). Per frame:

1. **Splat** — one `THREE.Points` draw of every clustered point into a
   half-float, half-resolution render target with additive blending. Each point
   is a Gaussian whose sigma is set per cluster in data units
   (`kernelFrac × projected rms`), converted to pixels by the projection in the
   vertex shader, so halos are world-space and consistent under zoom. The
   sprite diameter is clamped to `[minPointPx, maxPointPx]`; the minimum clamp
   is compensated in weight (brightness stays stable when zoomed out), the
   maximum clamp is not (haze fades as the camera enters a cluster).
2. **Halo** — a 17-tap separable Gaussian blur at quarter resolution, added back
   at `haloWeight`. Glow that spreads beyond the points, at a cost independent
   of point count.
3. **Resolve** — fullscreen quad: exposure → hue-preserving soft knee
   `1 − exp(−I^lift)` on the brightest channel → overlay canvas. Additive colour
   mixing is kept (two overlapping hues still give a vivid secondary colour);
   the knee means stacked clusters approach but never reach white. Light mode
   writes white pulled toward the haze hue, which the existing multiply blend
   turns into a tint of the cluster colour.

The brightness rule is explicit and unit-tested (`densityHazeMath.ts`):

```
sigma_c   = kernelFrac · rms2d_c
peak_c    = n_c · sigma_c² / (std_c² + sigma_c²)        expected accumulated peak, std_c = per-axis std (zoom-invariant)
weight_c  = peak_c ^ (−gamma)                            gamma 0 = raw density, 1 = every cluster equal
exposure  = exposureTarget / median_c(peak_c · weight_c)
```

`gamma` is the whole aesthetic decision. At 0 brightness is pure density
(dense = bright, diffuse = faint). At 1 every cluster peaks equally regardless
of spread. In between, diffuse clusters get a partial lift.

## What the replica predicts

Spread = 90th/10th percentile ratio of each cluster's haze on its own points
(lower = more even). Near-white = lit pixels with every channel ≥ 0.75.

| Dataset | Sprites | Density γ = 0.75 | Density γ = 0.5 |
|---|---|---|---|
| WordNet topics, spread | 2.6× | 1.8× | 3.4× |
| WordNet, near-white | 0% | 0.6% | 0% |
| Lacan subtopics, spread | 3.5× | 2.0× | 3.5× |
| Lacan, near-white | 0% | 2.4% | 0.1% |
| EMNLP topics, spread | 9.9× | 2.6× | 3.1× |
| trec fine labels, near-white | 25%, clipped core | 0% | 1.6% |

Visual verdicts from the side-by-side renders: WordNet at γ = 0.5 is the same
pointillist structure with brighter, cleaner glows. Lacan is the sensitive case:
at γ = 0.75 three or four large diffuse subtopics become frame-filling washes
that bury the fine dust; at γ ≤ 0.5 the dark field survives and the diffuse
subtopics stay as faint tints. EMNLP shows all 60 topics; emotion-by-label
becomes a mottled cloud showing local majorities instead of a grey fog; trec
fine labels go from a blown-out core to a bounded pastel cloud.

## Using it

Controls panel → View → **Nebula effects** → **Haze renderer** `Sprites | Density`.
With Density selected two sliders appear:

- **Haze strength** — multiplies `exposureTarget` (store `hazeStrength`, default 1×).
- **Equalise clusters** — `gamma` (store `hazeEqualize`, default 0.5).

All three persist with the other view preferences (`viz-preferences`). The
renderer kind is read in `ScatterPlot3D`'s haze lifecycle effect through
`createHazeRenderer` (`lib/utils/hazeRendering.ts`); the sliders retune the
live renderer through `setTuning` without a rebuild. The `Density` option is
disabled, and a persisted `density` preference reset to sprites, in browsers
without WebGL2 float render targets (`densityHaze/support.ts`; the one fallback
rule is `supportedHazeRenderer`, shared by the Controls panel, the renderer
factory and the per-collection overrides below).

**Per-collection defaults.** `COLLECTION_VIEW_OVERRIDES` in
`lib/utils/tourPresets.ts` can pin `hazeRenderer` per collection, applied once
per collection switch alongside its nebula / cluster-label flags (manual changes
stick until the next switch). The demo collections EMNLP
(`acl_abstracts_emnlp_findings`), `emotion` and the SAE label map
(`Gemma_9_16k_embedded`) open in density haze with cluster labels;
`wordnet_senses_full` pins sprites, so the density renderer does not follow a
visitor into the WordNet galaxy whose look and tour were tuned on sprites.
Where float targets are missing, the override applies sprites instead.

Constants not exposed in the UI live in `DEFAULT_DENSITY_HAZE_OPTIONS`
(`DensityHazeRenderer.ts`): `kernelFrac` 0.8, `lift` 0.8, `haloWeight` 0.5,
`haloBlurPx` 12, point clamps 2–32 px, `resolutionScale` 0.5.

## Test plan

- Unit: `lib/utils/__tests__/densityHazeMath.test.ts` (kernel, peak, γ-weight,
  exposure, clamp compensation, buffer layout, reweighting).
- Manual, each with the sprite renderer for comparison: WordNet `topic_label`
  (paper look), Lacan `subtopic_label` (diffuse subtopics must stay subdued),
  EMNLP `topic_label` (large topics visible), emotion `label` and trec
  `fine_label` (no white-out). Check zoom in/out stability, muting/filtering
  updates, light mode, and the screenshot export (composites the same canvas).
- Performance: `benchmarks/fps/` pass `3d-nebula` with each renderer.

## Known gaps / follow-ups

- The halo blur radius is screen-space, so it does not grow when zooming in
  (same behaviour as the 2D density overlay).
- The sprite renderer's grain (random sprite sizes) is gone; per-point kernel
  jitter or resolve-pass noise would bring it back if wanted.
- The distance-based fade of the sprite renderer is not reproduced; the
  maximum point clamp provides a milder equivalent.
- Light mode uses a simple inversion and has not been tuned.
- Replica scripts (`haze_sim.py`, `density_sim.py`) were run from a session
  scratchpad and are not part of the repo.
