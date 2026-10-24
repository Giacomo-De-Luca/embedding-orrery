/**
 * The Plotly internals the 4K screenshot export relies on (`lib/utils/plotCapture.ts`,
 * documentation/SCREENSHOT_EXPORT.md).
 *
 * 3D: the capture raises gl-plot3d's `pixelRatio`, calls `redraw()`, reads the
 * drawing buffer and puts the ratio back, all in one task. That works because
 * gl-plot3d sizes its canvas from the parent's CSS size × the ratio on every redraw
 * and hands the ratio to every object it draws (glyph sizes and line widths scale
 * with it). The scene runs against the recording GL stub through the harness.
 *
 * 2D: the capture raises the graph's `plotGlPixelRatio` around `Plotly.toImage`.
 * These tests read the fork's sources: toImage must copy the graph's config before
 * it returns, the clone's regl context must take its pixel ratio (which regl-scatter2d
 * multiplies point sizes by) from that config, and the exported SVG must place the
 * WebGL image at the CSS size, so the ratio sets its resolution.
 */
import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FakeCanvasElement, RecordingGl, StackglHarness, type GlScene } from './stackglHarness';

const { gl_plot3d: glPlot3d } = StackglHarness.load();
const frames = StackglHarness.frames;

const source = (file: string) => readFileSync(new URL(`../${file}`, import.meta.url), 'utf8');

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const PARENT = { clientWidth: 800, clientHeight: 500 };

/** A canvas inside a sized parent, so gl-plot3d's auto-resize has a CSS size to read. */
class SizedCanvas extends FakeCanvasElement {
  readonly style: Record<string, string> = {};
  readonly parentNode = PARENT;
}

/** An object that records the pixel ratio gl-plot3d gives it at each draw. */
class RatioRecorder {
  pickSlots = 1;
  pixelRatio = 1;
  dirty = false;
  axes: unknown = null;
  axesBounds: unknown = null;
  clipBounds: unknown = null;
  bounds = [[-1, -1, -1], [1, 1, 1]];
  readonly drawRatios: number[] = [];

  setPickBase(): void {}
  draw(): void {
    this.drawRatios.push(this.pixelRatio);
  }
  drawPick(): void {}
  pick(): null {
    return null;
  }
  highlight(): void {}
  isOpaque(): boolean {
    return true;
  }
  isTransparent(): boolean {
    return false;
  }
  dispose(): void {}
}

let scene: GlScene | null = null;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  frames.reset();
});

afterEach(() => {
  scene?.dispose();
  scene = null;
  vi.useRealTimers();
});

describe('gl-plot3d at an export pixel ratio', () => {
  it('redraws at a raised ratio in one call and returns to the live one', () => {
    const canvas = new SizedCanvas();
    const object = new RatioRecorder();
    scene = glPlot3d.createScene({
      gl: new RecordingGl().gl,
      canvas,
      cameraObject: { tick: () => false, matrix: IDENTITY.slice() },
      pixelRatio: 2,
      autoResize: true,
    });
    scene.add(object);
    frames.step();
    expect([canvas.width, canvas.height]).toEqual([1600, 1000]);
    object.drawRatios.length = 0;

    scene.pixelRatio = 4.8;
    scene.redraw();
    expect([canvas.width, canvas.height]).toEqual([3840, 2400]);
    expect(object.drawRatios).toEqual([4.8]);
    // The CSS size stays the parent's, so overlays laid out from it don't move
    expect([canvas.style.width, canvas.style.height]).toEqual(['800px', '500px']);

    scene.pixelRatio = 2;
    scene.redraw();
    expect([canvas.width, canvas.height]).toEqual([1600, 1000]);
    expect(object.drawRatios).toEqual([4.8, 2]);
  });
});

describe('Plotly 2D export internals', () => {
  it("toImage copies the graph's config synchronously, before its promise", () => {
    const toImage = source('plotly.js/src/plot_api/to_image.js');
    const copy = toImage.indexOf('var configImage = Lib.extendFlat({}, config, {');
    expect(toImage).toContain('config = gd._context;');
    expect(copy).toBeGreaterThan(-1);
    expect(copy).toBeLessThan(toImage.indexOf('return new Promise(function(resolve, reject) {\n        plotApi.newPlot(clonedGd'));
  });

  it("the regl context takes the config's plotGlPixelRatio, which scales regl-scatter2d's point sizes", () => {
    expect(source('plotly.js/src/lib/prepare_regl.js')).toContain('pixelRatio: gd._context.plotGlPixelRatio || global.devicePixelRatio');
    const scatter = source('../node_modules/regl-scatter2d/bundle.js');
    expect(scatter).toContain("pixelRatio: regl.context('pixelRatio')");
    expect(scatter).toContain('float pointSizeScale = (constPointSize) ? 2. : pixelRatio;');
    expect(source('plotly.js/src/traces/scattergl/convert.js')).not.toContain('constPointSize');
  });

  it('the exported SVG places the WebGL image at its CSS size', () => {
    const cartesian = source('plotly.js/src/plots/cartesian/index.js');
    expect(cartesian).toContain("var imageData = canvas.toDataURL('image/png');");
    expect(cartesian).toMatch(/width: canvas\.style\.width,\s*height: canvas\.style\.height/);
  });
});
