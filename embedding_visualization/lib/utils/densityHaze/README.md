# densityHaze — density-based nebula haze for ScatterPlot3D (experimental)

Alternative to the sprite-based `../hazeRenderer.ts`, selectable from the
Controls panel (Nebula effects → Haze renderer) and created through
`../hazeRendering.ts`. Design notes, replica results and tuning guidance:
`documentation/NEBULA_DENSITY_HAZE.md`.

## Files

| File | Responsibility |
|------|----------------|
| `densityHazeMath.ts` | Pure math, unit-tested (`../__tests__/densityHazeMath.test.ts`): per-cluster kernel sigma, expected density peak, γ-weight, global exposure, point-size clamp compensation, and `buildDensityHazeBuffers` / `reweightDensityHazeBuffers` (cluster map → typed attribute arrays). |
| `support.ts` | Three-free capability probe (`isDensityHazeSupported`, WebGL2 + `EXT_color_buffer_float`, cached) used by the renderer, the factory and the Controls panel gate. |
| `DensityHazeRenderer.ts` | Three.js renderer owning the GL resources: one `THREE.Points` splat pass into a half-float target, a quarter-resolution separable blur for the halo, and a resolve pass (exposure → hue-preserving soft knee). Same `updateClusters` / `resize` / `render` / `dispose` contract as `HazeRenderer`, plus `setTuning` for live γ/exposure changes and a static `isSupported()` probe. |

## Brightness rule

```
sigma_c   = kernelFrac · rms2d_c                 kernel per cluster (data units)
peak_c    = n_c · sigma_c² / (std_c² + sigma_c²)     expected accumulated peak (std_c = per-axis std), zoom-invariant
weight_c  = peak_c ^ (−gamma)                    gamma 0 = raw density, 1 = equalised clusters
exposure  = exposureTarget / median_c(peak_c · weight_c)
out       = hue-preserving knee( exposure · (acc + haloWeight · blur(acc)) )
```

The store exposes `hazeStrength` (multiplies `exposureTarget`) and
`hazeEqualize` (= gamma); both persist with the other view preferences.
