# Screenshot export

The camera button in both scatter plots' modebar downloads one PNG of the
visualization: the plot, its overlay canvases and the legend, on a black (dark
mode) or white (light mode) background. The image is 4K on its long side
(3840 px) whatever the screen, with the plot's on-screen aspect ratio; a screen
whose own resolution is higher keeps it. Code:
`embedding_visualization/lib/utils/plotCapture.ts`.

## Layers

| Layer | 2D | 3D |
|---|---|---|
| Plot | `Plotly.toImage` (draws the SVG axes and the WebGL points together) | the live GL scene, read back in one task (below) |
| Density / haze overlay | snapshot of the density canvas | snapshot of the haze canvas, drawn with `screen` (dark) / `multiply` (light) to match its CSS `mix-blend-mode` |
| Labels | cluster and search-result labels (`PLOT_LABELS.md`), redrawn on a canvas of their own | same |
| Legend | live DOM rasterized with `html-to-image` over a blurred backdrop (`drawLegendOverlay`) | same |

## Resolution

`ScreenshotScale.of(cssWidth, cssHeight, devicePixelRatio)`
(`lib/utils/screenshotScale.ts`, unit-tested) gives the export's pixels per CSS
pixel: `max(devicePixelRatio, 3840 / long side)`. A 1440 × 900 plot on a 1×
screen exports at 3840 × 2400, a 1280 × 720 one on a retina screen at
3840 × 2160, a 390 × 700 phone plot at 2139 × 3840. Each layer reaches that
resolution its own way:

The two WebGL ratio swaps live in `ExportPixelRatio`
(`lib/utils/exportPixelRatio.ts`, unit-tested with stand-ins that throw): each
raises the ratio for one call, never below the live one, and puts it back in a
`finally`.

- **2D plot.** `Plotly.toImage` scales the SVG by `scale`, but its clone draws
  the WebGL points at the graph's `plotGlPixelRatio` (2 by default), which would
  leave them upscaled. `ExportPixelRatio.plotToImage` raises
  `gd._context.plotGlPixelRatio` to the export scale for the call only: toImage
  copies the config before it returns, and regl-scatter2d multiplies point sizes
  by the ratio, so the points keep their on-screen size and stay sharp.
- **3D plot.** `ExportPixelRatio.redrawAt` sets gl-plot3d's `pixelRatio` to the
  export scale (never below the live 2), redraws, reads the drawing buffer and
  puts the live ratio back, all in one task. gl-plot3d sizes its canvas from the
  container's CSS size × the ratio on every redraw and scales glyphs and line
  widths with it; the CSS size doesn't change, so the overlays stay put, and the
  browser composites only after the task, so the export-size frame never reaches
  the screen. If the context is lost at the export size, the read throws
  ("Screenshot failed" toast) instead of saving a blank image.
- **Labels.** Each plot draws its labels with `drawLabelsOn(target)`;
  `renderLabels()` is the overlay's call (`LabelCanvas.overlay`, it also clears
  the click targets), and `LabelCanvas.renderOffscreen` runs `drawLabelsOn` on a
  fresh canvas at the export scale with empty click targets, so the text is drawn
  at 4K and the overlay is untouched. The 2D labels are drawn before `toImage`'s
  await, from the same layout it clones. A 3D fly-to hides the labels until it
  lands, so a screenshot taken during one leaves them out, as the screen does.
- **Legend.** `drawLegendOverlay` already rasterizes at the output canvas's
  width over the legend's CSS width.
- **Density and haze overlays are upscaled**, with `imageSmoothingQuality:
  'high'`. Both are soft fields (the 3D haze already renders at CSS resolution).
  Re-rendering the 2D density at 4K would need four float framebuffers of about
  130 MB each, and its blur works in device pixels, so it would also change the
  look. Its 1 px contour lines come out slightly soft.

The pins on Plotly and gl-plot3d internals are in
`embedding_visualization/forked/__tests__/screenshotScaleContract.test.ts`.

**Memory.** A 4K export allocates several full-size buffers for a moment. In 3D
the live context's drawing buffer is antialiased (multisampled, with depth), so
at 3840 × 2400 it takes a few hundred MB of GPU memory; the read-back, the
flattened copy, the GL image canvas, the label canvas and the output canvas take
about 37 MB each. In 2D the toImage clone gets a 4K WebGL context of its own, and
the plot goes through PNG encode/decode round trips at 4K. On phones this is the
largest change: a 390 × 700 plot goes from a 780 × 1400 drawing buffer to
2139 × 3840, about 7.5× the pixels. GPU memory on mobile Safari is unmeasured
(see the WordNet crash in `HF_SPACE_DEMO.md`); a lost context fails the export
with a toast and leaves the live 3D plot blank until a reload.

## 3D read-back and the search glow

`capture3DPlot` calls `glplot.redraw()` and reads the drawing buffer with
`gl.readPixels` in the same task. That works with `preserveDrawingBuffer: false`
and uses the live camera, which fly-to animations move without updating the
layout.

The pixels it reads are **premultiplied**, and not always valid ones.
gl-plot3d blends with `(ONE, ONE_MINUS_SRC_ALPHA)`, but gl-scatter3d's fragment
shader writes `colour × opacity` without multiplying RGB by the colour's own
alpha. A marker or line with an `rgba()` colour therefore leaves RGB above
alpha. The search-result glow layers (`rgba(…, 0.15)` and `rgba(…, 0.35)` at
trace opacity 0.15 / 0.3) and the connection lines are all like this. The
browser composites the canvas as `rgb + (1 − a) × backdrop`, so on screen these
pixels add their full colour: that is the glow.

Plotly's own `scene.toImage` converts to straight alpha (`rgb × 255 / a`,
clamped at 255). For these pixels that clamp throws most of the light away.
The dark outer glow, about (21, 27, 35) on screen, came out of the screenshot
at about (6, 6, 6), and the inner glow turned a flat grey. That is why
screenshots looked like they were missing the outer glow layers. The capture
used `scene.toImage` until 2026-10-22.

`PremultipliedPixels.flattenOnto` (`lib/utils/premultipliedPixels.ts`,
unit-tested) instead flips the rows and composites each pixel onto the export
background with the compositor's formula (`rgb + (1 − a) × background`, clamped
at 255). The result matches what the browser shows. Light mode was barely
affected, since glow pixels over white clamp to near-white either way.

The 2D path doesn't have this problem: regl-scatter2d blends with
`srcRGB: src alpha`, so its buffer always holds valid premultiplied colour, and
Plotly reads it with `toDataURL`. The haze canvases are opaque
(`alpha: false`), so a plain `drawImage` copies them exactly.
