# densityRenderer — 2D density overlay for ScatterPlot2D

WebGL2 port of Apple **embedding-atlas**'s density-mode rendering
(`references/embedding-atlas/packages/component/src/lib/webgl2_renderer/`,
MIT licensed — Apple copyright headers preserved in `shaders.ts`). Renders a
KDE-style density map with quantized color bands and Sobel iso-contours onto
an overlay canvas positioned over the Plotly plot area.

## Files

| File | Responsibility |
|------|----------------|
| `densityMath.ts` | Pure math (unit-tested, no GL): `Matrix3` helpers, `buildPositionMatrix` (Plotly axis ranges → clip space), `approximateMaxDensity2D` (peak density estimate, statistics.ts port), `computeViewingParams` (zoom-adaptive crossfade, `viewingParameters` port), `assignCategoryChannels` (category → RGBA channel, muted fallback), `buildColorMatrix` (gamma-linearized channel colors). |
| `shaders.ts` | GLSL ES 3.0 sources, near-verbatim from embedding-atlas: point splat (category / plain), R20 separable gaussian blur (+ pre-baked filter constants), density-band paint, contour paint, gamma correction. |
| `glResources.ts` | Raw-WebGL2 helpers (program/buffer/float-framebuffer), de-dataflowed from their `utils.ts`. |
| `DensityRenderer.ts` | The renderer class: `setData` → `setSize` → `render(props)` per frame; `dispose()`. Lifecycle/content split mirrors `lib/utils/hazeRenderer.ts`. |

## Pipeline (per frame)

1. **Splat**: every visible point drawn as a 1px `GL_POINTS` with additive
   blending into an RGBA32F count buffer. Three modes (`assignCategoryChannels`):
   - **categorical** (≤4 active categories): category index selects one of
     the 4 one-hot channels — exact per-category density, Apple's design;
   - **meanColor** (>4 categories): each point splats its category's
     linearized RGB + 1 into alpha, so RGB/A recovers the locally dominant
     cluster color — density bands *and* contour lines take that color;
   - **muted** (numeric / no color field): everything in channel 0, painted
     with a neutral tint.
2. **Blur**: separable gaussian, σ ≈ 20 device px, as 4 pre-baked filter
   passes per direction (8 draws, ping-pong FBOs). Blurred counts = KDE.
3. **Paint**: density quantized into bands (`quantizationStep`, default 0.1),
   up to 4 channel colors mixed per pixel; contours drawn per channel via a
   Sobel operator over the quantized density.
4. **Gamma**: linear→sRGB onto the canvas, cropping the blur safe margin
   (FBOs are oversized by 61px per side so blur has data past the visible
   edge).

Zoom behavior comes from `computeViewingParams`: density/contour alphas stay
at 1 through the overview and fade to 0 between `DENSITY_FADE_START_ZOOM`
(3×) and `DENSITY_FADE_END_ZOOM` (8×) relative zoom-in, leaving only Plotly's
own points. (Apple fades against an absolute density constant tuned for
multi-million-point datasets — near-invisible on 1k-150k collections.)

## Divergences from embedding-atlas

- **Transparent compositing**: Apple clears the linear buffer to opaque
  white/black because their renderer owns the whole view. We clear to
  transparent, blend the paint passes over it, and make the gamma pass
  alpha-preserving (unpremultiply → pow → re-premultiply) so the canvas
  composites over Plotly with normal source-over in both themes. In light
  mode the density shader already bakes the white base into lit pixels
  (`c = 1 − α + Σcᵢ`), so the result over the opaque white plot background
  matches Apple's. If dark-mode fringing ever shows up, the fallback lever is
  `mix-blend-mode: screen` on the canvas (dark only).
- **No point drawing**: Apple's density mode also disc-blurs and paints the
  points; Plotly already renders ours, so `disc_blur` / `paint_points` were
  not ported.
- **Dataflow harness dropped**: their `dataflow.ts` is a memoization/lifecycle
  graph with no algorithmic content; the class holds resources directly.
- `computeViewingParams` works from axis spans instead of a viewport scale;
  equivalence with Apple's scale-form formula is pinned by a unit test
  (`__tests__/densityMath.test.ts`).
- **Intensity boost**: Apple's `0.2` density scaler caps the densest band at
  ~20% opacity — tuned for a renderer that draws its own points. Over
  Plotly's points that is nearly invisible, so `DENSITY_INTENSITY = 3`
  multiplies the scaler by default (peak band ≈ 60%); pass `intensity: 1` to
  `computeViewingParams` for Apple's original look.

## Requirements

WebGL2 plus three extensions, gated by `DensityRenderer.isSupported()`:
`EXT_color_buffer_float` (render to float FBOs — load-bearing),
`EXT_float_blend` (additive blending into RGBA32F), and
`OES_texture_float_linear` (LINEAR sampling of float textures, used by the
blur's fractional taps). Unsupported environments simply don't get the
overlay; the scatter plot is unaffected.

## Consumers

`lib/hooks/useDensityOverlay.ts` owns the React lifecycle (construct/dispose,
data upload, rAF-coalesced renders on Plotly relayout) and is used by
`app/components/ScatterPlot2D.tsx` when the `densityMode` store flag is on.
