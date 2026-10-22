# Screenshot export

The camera button in both scatter plots' modebar downloads one PNG of the
visualization: the plot, its overlay canvases and the legend, at screen size ×
`devicePixelRatio`, on a black (dark mode) or white (light mode) background.
Code: `embedding_visualization/lib/utils/plotCapture.ts`.

## Layers

| Layer | 2D | 3D |
|---|---|---|
| Plot | `Plotly.toImage` (draws the SVG axes and the WebGL points together) | the live GL scene, read back in one task (below) |
| Density / haze overlay | snapshot of the density canvas | snapshot of the haze canvas, drawn with `screen` (dark) / `multiply` (light) to match its CSS `mix-blend-mode` |
| Labels | label canvas | label canvas |
| Legend | live DOM rasterized with `html-to-image` over a blurred backdrop (`drawLegendOverlay`) | same |

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
