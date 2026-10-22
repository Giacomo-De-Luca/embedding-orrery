# Pinch zoom on touch screens

Plotly has no two-finger gesture on touch screens. In 2D one finger pans and a
second finger is read as a new drag start (by Plotly's click-train logic, two
fingers lifted within 300 ms count as a double click, which resets the axes). In
3D gl-plot3d's camera reads only `changedTouches[0]`, so a second finger made the
rotation jump. Neither plot could be zoomed on a phone.

Both scatter plots now zoom with a pinch. One finger keeps its old behaviour.

## Behaviour

| | One finger | Two fingers |
|---|---|---|
| 2D | Plotly pan (unchanged) | Zoom about the fingers' midpoint |
| 3D | Plotly rotation (unchanged) | Camera dolly toward the centre, as the wheel does |

- The zoom follows the change in finger spread: fingers apart zoom in, together zoom out.
- Zoom-out limits are those of the wheel: 2D stops at twice the data extent
  (`useZoomLimit`), 3D at the camera's `zoomMax` (2, set in the Plotly fork).
- Once a second finger lands the gesture belongs to the pinch until every finger
  has lifted. A single finger left down after a pinch does nothing, because Plotly
  would resume its pan or rotation from where that finger was before the pinch and
  jump. While the gesture is the pinch's, any two fingers pinch: putting a finger
  back down starts a new pinch, and so does lifting one finger of the pair while
  two others remain.
- Fingers that started outside the plot (on a panel, the legend) are ignored.
- Both plot containers carry `touch-action: none`, so the browser never zooms or
  scrolls the page from the plot.
- A trackpad pinch on a laptop is a wheel event with `ctrlKey` and goes through
  Plotly's wheel zoom as before.

### Known gap: 2D on touch screens that can also hover

On a device whose primary pointer can hover (`(hover: none)` does not match:
touchscreen laptops, Chromebooks, probably an iPad with a trackpad), Plotly's
`dragelement` covers the whole page with a fixed `.dragcover` as soon as the first
finger starts a drag. A second finger then lands on that cover, outside the plot,
so the 2D pinch only starts when both fingers land in the same `touchstart`. Phones
report `(hover: none)`, get no cover and are not affected; neither is 3D, which
has no cover.

The fix is a one-line fork patch in `src/components/dragelement/index.js` that
skips the cover for touch drags (`if(hasHover && !rightClick && !e.touches)`; the
cover only matters for mouse drags, and `removeElement(undefined)` is already a
no-op). It is not applied: it would be logged in `forked/plotly_changes.md` with
a test in `forked/__tests__/`.

## Structure

| Piece | File | Role |
|---|---|---|
| `PinchGesture` | `embedding_visualization/lib/utils/pinchGesture.ts` | DOM-free recogniser. Per touch event it returns whether to keep the event from Plotly, the pair's frame (spread ÷ start spread, midpoint), and whether a pinch started or ended. |
| `PinchZoomTarget` | `embedding_visualization/lib/utils/plotPinchZoom.ts` | `start(frame)`, `pinch(frame)`, `end()`, implemented once per plot type. |
| `PinchZoomSession` | same | DOM-free event handling: feeds `PinchGesture`, drives the target, coalesces moves to one `pinch` per animation frame (scheduler injected), applies a waiting frame before `end` and nothing after. |
| `CartesianPinchZoom` | same | 2D target, see below. |
| `CameraPinchZoom` | same | 3D target, see below. |
| `usePinchZoom` | `embedding_visualization/lib/hooks/usePinchZoom.ts` | Capture-phase `touchstart`/`touchmove`/`touchend`/`touchcancel` listeners on the plot container. Maps `TouchEvent.touches` (fingers on this plot only) to the session and stops the events it consumes before Plotly. |

`ScatterPlot2D` and `ScatterPlot3D` each create their target once
(`useState(() => new …(graphDivRef))`) and call `usePinchZoom(containerRef, target)`.

### 2D: replayed as wheel events

`CartesianPinchZoom` turns each frame's spread change into `wheel` events on
Plotly's main drag surface (`.nsewdrag`), at the fingers' midpoint. That reuses
Plotly's wheel zoom (`zoomWheel` in `forked/plotly.js/src/plots/cartesian/dragbox.js`):

- each event updates only the scattergl viewport (no buffer upload);
- one GUI relayout commits the ranges once the gesture pauses for 50 ms, so
  `uirevision` keeps the zoom across later `Plotly.react` calls;
- `plotly_relayouting` fires per event, so the density map and cluster labels follow;
- the events pass the container's capture-phase `useZoomLimit` listener on the way
  down, so the 2D zoom-out limit needs no second implementation.

`zoomWheel` multiplies the axis span by `exp(deltaY / 200)` and clamps `|deltaY|`
to 20 per event. `CartesianPinchZoom.wheelDeltas(spanRatio)` inverts that and
splits large ratios into several unclamped events. Typical pinch speeds need one
event per frame; a fast pinch needs two or three, each a full Plotly viewport
redraw. `forked/__tests__/pinchZoomContract.test.ts` checks the formula and the
internals below against the fork's sources, so a fork patch that changes them
fails a test.

Plotly started a one-finger drag when the first finger landed. `start` hands it
over to the pinch:

- `gd._dragging = false`: at the final touchend Plotly's `onDone` only removes its
  document listeners. Without this it would fire a click (or a double click, which
  autoranges) or commit a stale pan.
- `delete gd._dragdata`: every relayout runs `Plots.redrag`, which would otherwise
  replay the old pan offset on top of the new ranges.
- If Plotly had already panned (`gd._dragged`), `end` sends one zero-delta wheel so
  Plotly commits that pan through the same relayout.

### 3D: camera distance

`CameraPinchZoom` takes the camera's distance when the pinch starts and sets
`camera.distance` to `start / scale` on each frame. The camera's filtered radius
clamps every value to its distance limits, which is how the wheel is capped too.

The `distance` getter returns the value of the scene's last render, which lags a
write by a frame or two. `start` therefore reads the view instead, after bringing
it up to its latest keyframe (`view.recalcMatrix(view.lastT())`, as Plotly's
`scene.getCamera` does), so a pinch that restarts right after another starts
where that one left the camera.

No relayout is sent: the scene redraws from its own render loop, the haze and
label overlays follow the camera as they do for rotation, and `ScatterPlot3D`
copies the live camera into the layout before every `Plotly.react`, so the zoom
survives re-renders. An orthographic camera would ignore the distance (the wheel
changes its aspect ratio instead); the app only uses perspective.

## Tests

- `lib/utils/__tests__/pinchGesture.test.ts`: one finger never consumed; pinch
  start, scale and midpoint; a third finger ignored; lifting one finger ends the
  pinch and swallows the finger left behind; the last lift released; a pinch
  restarting when a finger lands again, or on the two fingers left beside a third;
  no pinch starting mid-move; coincident fingers.
- `lib/utils/__tests__/plotPinchZoom.test.ts`:
  - `wheelDeltas` reproduces the span ratio through Plotly's formula and never
    exceeds the clamp;
  - the 2D target replays per-frame spread changes at the midpoint, hands over
    Plotly's drag and commits an earlier pan;
  - the 3D target scales from each pinch's start, leaves the limit to the camera,
    and starts from the latest distance before the camera has rendered it;
  - both targets do nothing before the plot exists;
  - `PinchZoomSession`: one finger untouched, start on the second finger, one
    `pinch` per frame with the latest spread, a waiting frame applied before `end`
    and nothing after, `dispose` dropping a waiting frame.
- `forked/__tests__/pinchZoomContract.test.ts`: the Plotly internals above, read
  from the fork's sources.

The browser plumbing in `usePinchZoom` and Plotly's response are not covered by
unit tests (the suite runs in node). Check on a phone or a touch emulator:

1. 2D and 3D: pinch in and out, zoom tracks the fingers smoothly.
2. 2D: zoom out stops at the limit; zoom stays after recolouring or rotating the phone.
3. 2D: drag one finger, add a second, pinch, lift both: no jump, no point selected, no reset.
4. 3D: rotate with one finger, add a second, pinch, lift one, move the other: no rotation jump.
5. 3D: zoom stays after recolouring; density haze and labels follow the zoom.
6. Tap a point after a pinch: it still selects.
