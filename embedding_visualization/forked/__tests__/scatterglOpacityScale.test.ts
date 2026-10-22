/**
 * Fork change 6 (`../plotly_changes.md`): the `scatterglOpacityScale` config
 * option scales each scattergl trace's marker opacity before every draw, which is
 * how the 2D plot raises the opacity while zooming in (`lib/utils/zoomOpacity.ts`).
 *
 * The scene tests run the fork's real `scene_update.js` with a stand-in for the
 * regl-scatter2d instance that records the opacity of each group as it is drawn.
 * The contract tests read sources the patch relies on: regl-scatter2d taking the
 * opacity uniform from the group at draw time, and `toImage` handing the graph's
 * config to its clone, which is what makes screenshots match the screen.
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

type OpacityScale = (trace: unknown, xRange: number[], yRange: number[]) => number;

/** A scattergl scene holding one marker trace per entry; `null` is a trace without markers. */
function buildScene(markerOpacities: (number | null)[], scale?: OpacityScale) {
  const gd = { _context: { scatterglOpacityScale: scale ?? null }, calcdata: [] as unknown[] };
  const subplot: { _scene?: any } = {};
  sceneUpdate(gd, subplot);
  const scene = subplot._scene;

  const drawn: { index: number; opacity: number }[] = [];
  const groups = markerOpacities.map((opacity) => ({ opacity: opacity ?? 1 }));
  scene.scatter2d = {
    groups,
    draw: (index: number) => drawn.push({ index, opacity: groups[index].opacity }),
  };

  const traces = markerOpacities.map((opacity, index) => {
    scene.markerOptions.push(opacity === null ? null : { opacity });
    scene.selectBatch.push([]);
    scene.unselectBatch.push([]);
    const trace = { index, _xA: { range: [0, 10] }, _yA: { range: [5, 15] } };
    gd.calcdata.push([{ trace, t: { _scene: scene, index } }]);
    return trace;
  });
  scene.count = markerOpacities.length;

  return { scene, groups, traces, drawn };
}

describe('scatterglOpacityScale config option', () => {
  it('is declared with a null default, which is what lets setPlotContext copy it into the graph context', () => {
    expect(configAttributes.scatterglOpacityScale.valType).toBe('any');
    expect(dfltConfig.scatterglOpacityScale).toBeNull();
  });

  it('scales every marker group from its base opacity before it is drawn', () => {
    const { scene, drawn } = buildScene([0.15, 0.3], () => 2);
    scene.draw();
    expect(drawn).toEqual([{ index: 0, opacity: 0.3 }, { index: 1, opacity: 0.6 }]);
  });

  it('never compounds across draws', () => {
    const { scene, drawn } = buildScene([0.15], () => 2);
    scene.draw();
    scene.draw();
    expect(drawn.map((d) => d.opacity)).toEqual([0.3, 0.3]);
  });

  it("passes each trace with its axes' current ranges", () => {
    const scale = vi.fn<OpacityScale>(() => 1);
    const { scene, traces } = buildScene([0.15, 0.3], scale);
    scene.draw();
    expect(scale.mock.calls).toEqual([
      [traces[0], [0, 10], [5, 15]],
      [traces[1], [0, 10], [5, 15]],
    ]);
  });

  it('leaves traces with a selection, and traces without markers, to plotly', () => {
    const scale = vi.fn<OpacityScale>(() => 2);
    const { scene, groups, traces } = buildScene([0.15, 0.3, null], scale);
    scene.selectBatch[1] = [0];
    scene.draw();
    expect(groups.map((g) => g.opacity)).toEqual([0.3, 0.3, 1]);
    expect(scale.mock.calls.map(([trace]) => trace)).toEqual([traces[0]]);
  });

  it('keeps the opacity plotly set when the option is not given', () => {
    const { scene, drawn } = buildScene([0.15]);
    scene.draw();
    expect(drawn).toEqual([{ index: 0, opacity: 0.15 }]);
  });
});

describe('internals the option relies on', () => {
  it('regl-scatter2d reads the opacity uniform from the group when drawing', () => {
    // The patch writes group.opacity directly instead of calling update()
    const scatter = source('../../node_modules/regl-scatter2d/bundle.js');
    expect(scatter).toContain("opacity: regl.prop('opacity')");
    expect(scatter).toContain('if (!(group && group.count && group.opacity)) return;');
  });

  it("toImage plots its clone with the graph's config, so screenshots are scaled too", () => {
    const toImage = source('../plotly.js/src/plot_api/to_image.js');
    expect(toImage).toContain('config = gd._context;');
    expect(toImage).toContain('var configImage = Lib.extendFlat({}, config, {');
  });
});
