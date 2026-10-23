/**
 * Fork change 6 (`../plotly_changes.md`): the `scatterglOpacityScale` and
 * `scatterglSizeScale` config options scale each scattergl trace's marker opacity
 * and size before every draw, which is how the 2D plot makes points more opaque and
 * larger while zooming in (`lib/utils/zoomMarkerStyle.ts`).
 *
 * The scene tests run the fork's real `scene_update.js` with a stand-in for the
 * regl-scatter2d instance that records each group's opacity and size as it is drawn,
 * and the size updates sent through `update()`. The contract tests read sources the
 * patch relies on: regl-scatter2d taking the opacity uniform and a single size from
 * the group at draw time, `update()` leaving `undefined` entries alone, and `toImage`
 * handing the graph's config to its clone, which is what makes screenshots match
 * the screen.
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { describe, expect, it, vi } from 'vitest';

// Plotly's lib reads `self` (d3) and `window` (isPlainObject) when it loads
Object.assign(globalThis, { self: globalThis, window: globalThis });
const require = createRequire(import.meta.url);
const sceneUpdate = require('../plotly.js/src/traces/scattergl/scene_update.js');
const { configAttributes, dfltConfig } = require('../plotly.js/src/plot_api/plot_config.js');

const source = (file: string) => readFileSync(new URL(file, import.meta.url), 'utf8');

type MarkerScale = (trace: unknown, xRange: number[], yRange: number[]) => number;

/** One marker trace: a single size, per-point sizes, or `null` for a trace without markers. */
type TraceSpec = { opacity: number; size?: number; sizes?: number[] } | null;

interface Group {
  opacity: number;
  size: number | number[];
}

/** A scattergl scene holding one trace per spec, drawn by a recording stand-in for regl-scatter2d. */
function buildScene(specs: TraceSpec[], scales: { opacity?: MarkerScale; size?: MarkerScale } = {}) {
  const gd = {
    _context: { scatterglOpacityScale: scales.opacity ?? null, scatterglSizeScale: scales.size ?? null },
    calcdata: [] as unknown[],
  };
  const subplot: { _scene?: any } = {};
  sceneUpdate(gd, subplot);
  const scene = subplot._scene;

  const groups: Group[] = specs.map((spec) => ({ opacity: spec?.opacity ?? 1, size: spec?.sizes ?? spec?.size ?? 10 }));
  const drawn: { index: number; opacity: number; size: number | number[] }[] = [];
  const sizeUpdates: (Partial<Group> | undefined)[][] = [];
  scene.scatter2d = {
    groups,
    // A group index, or per-group point batches for a trace with a selection
    draw: (arg: number | number[][]) => {
      const indices = typeof arg === 'number' ? [arg] : arg.flatMap((points, i) => (points.length ? [i] : []));
      for (const index of indices) drawn.push({ index, opacity: groups[index].opacity, size: groups[index].size });
    },
    // Like regl-scatter2d: one options entry per group, `undefined` leaves a group as it is
    update: (options: (Partial<Group> | undefined)[]) => {
      sizeUpdates.push(options);
      options.forEach((option, i) => {
        if (option?.size !== undefined) groups[i].size = option.size;
      });
    },
  };

  const traces = specs.map((spec, index) => {
    scene.markerOptions.push(spec);
    scene.selectBatch.push([]);
    scene.unselectBatch.push([]);
    const trace = { index, _xA: { range: [0, 10] }, _yA: { range: [5, 15] } };
    gd.calcdata.push([{ trace, t: { _scene: scene, index } }]);
    return trace;
  });
  scene.count = specs.length;

  return { scene, groups, traces, drawn, sizeUpdates };
}

describe('marker scale config options', () => {
  it('are declared with a null default, which is what lets setPlotContext copy them into the graph context', () => {
    for (const option of ['scatterglOpacityScale', 'scatterglSizeScale']) {
      expect(configAttributes[option].valType).toBe('any');
      expect(dfltConfig[option]).toBeNull();
    }
  });

  it('scale every marker group from its base opacity and size before it is drawn', () => {
    const { scene, drawn } = buildScene(
      [{ opacity: 0.15, size: 3 }, { opacity: 0.3, size: 4 }],
      { opacity: () => 2, size: () => 1.5 },
    );
    scene.draw();
    expect(drawn).toEqual([
      { index: 0, opacity: 0.3, size: 4.5 },
      { index: 1, opacity: 0.6, size: 6 },
    ]);
  });

  it('never compound across draws', () => {
    const { scene, drawn } = buildScene([{ opacity: 0.15, size: 3 }], { opacity: () => 2, size: () => 1.5 });
    scene.draw();
    scene.draw();
    expect(drawn.map(({ opacity, size }) => [opacity, size])).toEqual([[0.3, 4.5], [0.3, 4.5]]);
  });

  it("pass each trace with its axes' current ranges", () => {
    const opacity = vi.fn<MarkerScale>(() => 1);
    const size = vi.fn<MarkerScale>(() => 1);
    const { scene, traces } = buildScene([{ opacity: 0.15, size: 3 }, { opacity: 0.3, size: 3 }], { opacity, size });
    scene.draw();
    const expected = [
      [traces[0], [0, 10], [5, 15]],
      [traces[1], [0, 10], [5, 15]],
    ];
    expect(opacity.mock.calls).toEqual(expected);
    expect(size.mock.calls).toEqual(expected);
  });

  it('leave the opacity of traces with a selection to plotly, and skip traces without markers', () => {
    const opacity = vi.fn<MarkerScale>(() => 2);
    const { scene, groups, traces } = buildScene(
      [{ opacity: 0.15, size: 3 }, { opacity: 0.3, size: 3 }, null],
      { opacity, size: () => 2 },
    );
    scene.selectBatch[1] = [0];
    scene.draw();
    expect(groups).toEqual([{ opacity: 0.3, size: 6 }, { opacity: 0.3, size: 6 }, { opacity: 1, size: 10 }]);
    expect(opacity.mock.calls.map(([trace]) => trace)).toEqual([traces[0]]);
  });

  it('grow both the unselected and the selected points of a trace with a selection', () => {
    const { scene, groups } = buildScene(
      [{ opacity: 0.15, size: 3 }, { opacity: 1, sizes: [3, 6] }],
      { opacity: () => 2, size: () => 2 },
    );
    // Selected points are drawn by select2d, from the selected style's size if it sets one
    const selectedGroups: Group[] = [{ opacity: 0.15, size: 4 }, { opacity: 1, size: [3, 6] }];
    scene.select2d = { groups: selectedGroups, draw: () => undefined };
    scene.markerSelectedOptions.push({ size: 4 }, {});
    scene.draw();
    // What select.js does: plotly's unselected opacity (opacity only), then the selection batches
    groups[0].opacity = 0.03;
    groups[1].opacity = 0.2;
    scene.selectBatch[0] = [0];
    scene.unselectBatch[1] = [1];
    scene.draw();
    expect(groups[0]).toEqual({ opacity: 0.03, size: 6 });
    expect(selectedGroups[0]).toEqual({ opacity: 0.15, size: 8 });
    // Per-point sizes have no select2d counterpart to scale, so a selection puts them back to base
    expect(groups[1]).toEqual({ opacity: 0.2, size: [3, 6] });
    expect(groups[1].size).toBe(scene.markerOptions[1].sizes);
  });

  it('keep the style plotly set when the options are not given', () => {
    const { scene, drawn, sizeUpdates } = buildScene([{ opacity: 0.15, size: 3 }, { opacity: 0.3, sizes: [3, 6] }]);
    scene.draw();
    expect(drawn).toEqual([{ index: 0, opacity: 0.15, size: 3 }, { index: 1, opacity: 0.3, size: [3, 6] }]);
    expect(sizeUpdates).toEqual([]);
  });

  describe('per-point sizes', () => {
    it('are uploaded through update(), with only the scaled group set', () => {
      const { scene, drawn, sizeUpdates } = buildScene(
        [{ opacity: 0.15, size: 3 }, { opacity: 1, sizes: [3, 6] }],
        { size: () => 2 },
      );
      scene.draw();
      expect(sizeUpdates).toEqual([[undefined, { size: [6, 12] }]]);
      expect(drawn[1]).toEqual({ index: 1, opacity: 1, size: [6, 12] });
    });

    it('are uploaded again only when the factor changes', () => {
      let factor = 2;
      const { scene, sizeUpdates } = buildScene([{ opacity: 1, sizes: [3, 6] }], { size: () => factor });
      scene.draw();
      scene.draw();
      expect(sizeUpdates).toHaveLength(1);
      factor = 3;
      scene.draw();
      expect(sizeUpdates).toEqual([[{ size: [6, 12] }], [{ size: [9, 18] }]]);
    });

    it('are scaled again after plotly resets the group to its base sizes', () => {
      const { scene, groups, sizeUpdates } = buildScene([{ opacity: 1, sizes: [3, 6] }], { size: () => 2 });
      scene.draw();
      // What plot.js does on every plot call: scatter2d.update(scene.markerOptions)
      groups[0].size = scene.markerOptions[0].sizes;
      scene.draw();
      expect(sizeUpdates).toHaveLength(2);
      expect(groups[0].size).toEqual([6, 12]);
    });

    it('are put back to the base sizes, once, when the factor returns to 1', () => {
      let factor = 2;
      const { scene, groups, sizeUpdates } = buildScene([{ opacity: 1, sizes: [3, 6] }], { size: () => factor });
      scene.draw();
      factor = 1;
      scene.draw();
      scene.draw();
      expect(sizeUpdates).toEqual([[{ size: [6, 12] }], [{ size: [3, 6] }]]);
      expect(groups[0].size).toBe(scene.markerOptions[0].sizes);
    });

    it('are not uploaded at all while the factor is 1', () => {
      const { scene, sizeUpdates } = buildScene([{ opacity: 1, sizes: [3, 6] }], { size: () => 1 });
      scene.draw();
      expect(sizeUpdates).toEqual([]);
    });
  });
});

describe('internals the options rely on', () => {
  const scatter = source('../../node_modules/regl-scatter2d/bundle.js');

  it('regl-scatter2d reads the opacity uniform and a single size from the group when drawing', () => {
    // The patch writes group.opacity and a single group.size directly instead of calling update()
    expect(scatter).toContain("opacity: regl.prop('opacity')");
    expect(scatter).toContain('if (!(group && group.count && group.opacity)) return;');
    expect(scatter).toContain('return prop.size.length ? {');
    expect(scatter).toContain('constant: [Math.round(prop.size * 255 / _this.maxSize)]');
  });

  it('regl-scatter2d update() leaves groups whose options are undefined untouched', () => {
    expect(scatter).toContain('if (options === undefined) return group;');
  });

  it('regl-scatter2d stores per-point sizes as Uint8 fractions of a 100 px maximum', () => {
    // Larger per-point sizes wrap; single sizes are a float constant and can't (the app only uses those)
    expect(scatter).toContain('maxSize = 100;');
    expect(scatter).toContain('sizes[_i2 * 2] = Math.round((size[_i2] == null ? size : size[_i2]) * 255 / maxSize);');
  });

  it("toImage plots its clone with the graph's config, so screenshots are scaled too", () => {
    const toImage = source('../plotly.js/src/plot_api/to_image.js');
    expect(toImage).toContain('config = gd._context;');
    expect(toImage).toContain('var configImage = Lib.extendFlat({}, config, {');
  });
});
