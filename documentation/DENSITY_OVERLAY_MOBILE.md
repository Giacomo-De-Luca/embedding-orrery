# 2D density overlay on phones

The 2D density overlay (Controls → View → Density overlay, and the main tour's
"flat map" step) drew nothing on phones. This page records why and the
half-float fallback that fixed it.

## Symptom

On an iPhone (and, going by the extension support below, on Android phones
whose GPUs lack 32-bit float filtering) the "Density overlay" checkbox ticked
and its intensity slider appeared, but no density bands or contours were
drawn. The 90-second tour, which runs on phones, reached its
density step and showed a bare 2D map.

## Cause

The overlay is a port of embedding-atlas's renderer
(`embedding_visualization/lib/utils/densityRenderer/`). It splats every point
additively into float framebuffers, blurs them with LINEAR-filtered taps and
paints the result. embedding-atlas uses RGBA32F buffers, which need three
WebGL2 extensions:

| Extension | Used for | Phones |
|---|---|---|
| `EXT_color_buffer_float` | rendering into float textures | available |
| `EXT_float_blend` | additive blending into **32-bit** float targets | missing on Apple GPUs (WebKit: Apple GPUs don't blend RGBA32F) |
| `OES_texture_float_linear` | LINEAR sampling of **32-bit** float textures | missing on iPhones (WebKit: the hardware can't filter RGBA32F; older iOS reported the extension and sampled NEAREST); about half of iOS devices in web3dsurvey, and some Android GPUs |

`DensityRenderer.isSupported()` required all three, so it returned false on
phones and `ScatterPlot2D` never mounted the overlay canvas. Nothing told the
user: the checkbox had no unsupported state.

## Fix

**Half-float buffers where 32-bit ones can't blend or filter.** In WebGL2,
RGBA16F textures are LINEAR-filterable in core, and `EXT_float_blend` only
concerns 32-bit targets, so `EXT_color_buffer_float` alone is enough. The 3D
density haze already renders into half floats with exactly that requirement.

- `lib/utils/floatRenderTargets.ts`: `FloatRenderTargets`, shared by both
  density renderers. `enable(gl)` enables the extensions and returns
  `float32` (all three present: desktops, unchanged), `float16`
  (`EXT_color_buffer_float` only: phones) or null. `probe()` / `available()`
  cache the answer from one throwaway context, which is then released
  (Safari caps live WebGL contexts).
- `DensityRenderer` takes its precision from `enable` on its own context, and
  `createFramebuffer` (`glResources.ts`) allocates R16F/RGBA16F with
  `HALF_FLOAT` at `float16`.
- `densityHaze/support.ts` (`isDensityHazeSupported`) now delegates to the
  same probe instead of keeping its own copy.

**No dead controls where even half floats are missing.**

- `VisualizationControls`: the "Density overlay" checkbox (and, as before, the
  3D haze's "Density" option) is disabled with a tooltip, both gated by
  `useFloatRenderTargets` (`lib/hooks/`, a `useSyncExternalStore` read of the
  cached probe, false during SSR). A persisted `densityMode: true` is reset on
  mount, next to the existing haze-renderer reset (the store rehydrates before
  this effect runs: `StoreHydrator` is an earlier sibling).
- `mainTourSteps(densityOverlaySupported)` (`lib/utils/tourSteps.ts`) drops the
  tour's density step; `page.tsx` passes `DensityRenderer.isSupported()`.

## Trade-offs of float16

- **Count saturation**: half floats count exactly up to 2048. A device pixel
  receiving more than 2048 points stops counting, so the densest peak flattens
  (top band and its contour can merge). In mean-colour mode the summed colours
  lose precision earlier on such stacks. Only exact or near-duplicate
  positions reach this; desktops keep float32 and are unaffected.
- **Memory**: half the size. For an iPhone-sized plot at DPR 3 (about
  1180 × 2100 device px plus the 61 px blur margin per side) the four
  framebuffers take about 90 MB instead of about 185 MB.
- Blurred values (points per device pixel) stay in half-float's normal range
  for any cluster dense enough to reach the first band; only the faint tails
  below it lose precision.

## Tests

- `lib/utils/__tests__/floatRenderTargets.test.ts`: precision per extension
  combination, extensions enabled on the given context, probe caching, context
  release, no WebGL2, throwing context creation, SSR.
- `lib/utils/densityRenderer/__tests__/glResources.test.ts`: internal format,
  format and texel type per precision and channel count; LINEAR filtering for
  both.
- `lib/utils/__tests__/tourSteps.test.ts` (`mainTourSteps`): full tour with the
  overlay, the same order minus `density` without it, one stable array per
  answer (a fresh array would re-sync joyride's steps on every page render).
- `lib/utils/__tests__/densityHazeSupport.test.ts` still passes against the
  shared probe.

## On-device check

Not covered by unit tests (GPU behaviour):

1. iPhone Safari: Explore → Controls → 2D → Density overlay. Bands and contours
   draw; pinch and pan redraw them.
2. The 90-second tour on a phone reaches "The flat map" with contours.
3. Desktop: the overlay looks unchanged (still float32).
4. To see which tier a device gets: `webglreport.com/?v=2` lists the three
   extensions.

## Sources

- WebKit bugs 247240, 252359, 212005 (RGBA32F blending and filtering on Apple
  GPUs; `OES_texture_float_linear` misreported on iOS).
- web3dsurvey, `OES_texture_float_linear` support by platform.
