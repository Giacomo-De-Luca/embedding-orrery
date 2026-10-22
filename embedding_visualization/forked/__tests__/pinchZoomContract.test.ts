/**
 * The Plotly internals the pinch zoom relies on (`lib/utils/plotPinchZoom.ts`,
 * documentation/MOBILE_PINCH_ZOOM.md).
 *
 * The 2D pinch replays Plotly's wheel zoom and takes over Plotly's one-finger drag;
 * the 3D pinch writes gl-plot3d's camera distance. None of that is public API, so
 * these tests read the fork's sources: a fork patch that changes one of these
 * fails here instead of the pinch silently zooming at the wrong speed, firing a
 * click after a pinch, or replaying an old pan.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { CartesianPinchZoom } from '../../lib/utils/plotPinchZoom';

const source = (file: string) => readFileSync(new URL(`../plotly.js/${file}`, import.meta.url), 'utf8');

describe('Plotly internals behind the pinch zoom', () => {
  it('2D wheel zoom still scales the span by exp(clamped deltaY / scale) on the main drag surface', () => {
    const dragbox = source('src/plots/cartesian/dragbox.js');
    const { MAX_WHEEL_DELTA: max, WHEEL_ZOOM_SCALE: scale } = CartesianPinchZoom;
    expect(dragbox).toContain('var wheelDelta = -e.deltaY;');
    expect(dragbox).toContain(`Math.exp(-Math.min(Math.max(wheelDelta, -${max}), ${max}) / ${scale})`);
    expect(dragbox).toContain("mainplot.draglayer.select('.nsewdrag')");
    expect(dragbox).toContain('attachWheelEventHandler(dragger, zoomWheel)');
  });

  it('a 2D drag with _dragging cleared ends without a click, and relayouts replay only _dragdata', () => {
    const dragelement = source('src/components/dragelement/index.js');
    expect(dragelement).toMatch(/if\(!gd\._dragging\) \{\s*gd\._dragged = false;\s*return;\s*\}/);
    expect(dragelement).toContain('delete gd._dragdata;');
    expect(source('src/plots/cartesian/dragbox.js')).toMatch(/_redrag = function\(\) \{\s*var dragDataNow = gd\._dragdata;/);
  });

  it('the 3D camera distance setter still goes through the view, whose radius is clamped to the zoom limits', () => {
    const stackgl = source('stackgl_modules/index.js');
    expect(stackgl).toMatch(/set: function\(d\) \{\s*view\.setDistance\(view\.lastT\(\), d\)/);
    expect(stackgl).toMatch(/this\.computedRadius = this\._active\.computedRadius/);
    expect(source('src/plots/gl3d/scene.js')).toContain('zoomMax: 2');
  });
});
