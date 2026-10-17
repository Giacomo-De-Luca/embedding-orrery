import assert from 'node:assert/strict';
import test from 'node:test';

import { BenchPass, BenchRunConfig } from './lib/benchmarkContract.mjs';
import { GlSceneStats } from './lib/glSceneStats.mjs';

test('bench passes map to view mode, colouring and overlays', () => {
  assert.deepEqual(BenchPass.parse('3d'), {
    name: '3d', mode: '3d', nebula: false, colorTopics: false, densityMode: false,
  });
  assert.deepEqual(BenchPass.parse('3d-topics'), {
    name: '3d-topics', mode: '3d', nebula: false, colorTopics: true, densityMode: false,
  });
  assert.deepEqual(BenchPass.parse('3d-nebula'), {
    name: '3d-nebula', mode: '3d', nebula: true, colorTopics: true, densityMode: false,
  });
  assert.deepEqual(BenchPass.parse('2d'), {
    name: '2d', mode: '2d', nebula: false, colorTopics: false, densityMode: false,
  });
  assert.deepEqual(BenchPass.parse('2d-topics'), {
    name: '2d-topics', mode: '2d', nebula: false, colorTopics: true, densityMode: false,
  });
  assert.deepEqual(BenchPass.parse('2d-density'), {
    name: '2d-density', mode: '2d', nebula: false, colorTopics: false, densityMode: true,
  });
});

test('a missing pass name runs the 3d pass', () => {
  assert.equal(BenchPass.parse(undefined).name, '3d');
});

test('an unknown pass name is rejected instead of silently running 3d', () => {
  assert.throws(() => BenchPass.parse('2d-topic'), /unknown pass "2d-topic"/);
  assert.throws(() => BenchPass.parse('emotion'), /unknown pass/);
});

test('device scale factor parses positive numbers and defaults to 1', () => {
  assert.equal(BenchRunConfig.deviceScaleFactor(undefined), 1);
  assert.equal(BenchRunConfig.deviceScaleFactor('2'), 2);
  assert.equal(BenchRunConfig.deviceScaleFactor('1.5'), 1.5);
  assert.equal(BenchRunConfig.deviceScaleFactor('0'), 1);
  assert.equal(BenchRunConfig.deviceScaleFactor('abc'), 1);
});

test('result labels carry the device scale factor unless it is 1', () => {
  assert.equal(BenchRunConfig.resultLabel('baseline', 1), 'baseline');
  assert.equal(BenchRunConfig.resultLabel('baseline', 2), 'baseline-dpr2');
  // No dot, so the label and its file name match (artifact names replace dots).
  assert.equal(BenchRunConfig.resultLabel('current', 1.5), 'current-dpr1p5');
});

/** Installs a fake `document` whose plot div is `gd`, runs fn, then removes it. */
function withPlot(gd, fn) {
  globalThis.document = { querySelector: (sel) => (sel === '.js-plotly-plot' ? gd : null) };
  try {
    return fn();
  } finally {
    delete globalThis.document;
  }
}

const canvas = (width, height, cssWidth, cssHeight) => ({
  width, height, clientWidth: cssWidth, clientHeight: cssHeight,
});

test('3d stats sum fill and outline vertices of point clouds only', () => {
  const gd = {
    data: [{}, {}, {}],
    _fullLayout: {
      scene: {
        _scene: {
          glplot: {
            canvas: canvas(3200, 1900, 1600, 950),
            objects: [
              { vertexCount: 600, lineVertexCount: 400 },
              { vertexCount: 300, lineVertexCount: 0 },
              { vertexCount: 50 }, // a line plot: no lineVertexCount
            ],
            pickStats: { renders: 2, deferredFrames: 5, readbacks: 4, buffers: 2 },
          },
        },
      },
    },
  };
  const stats = withPlot(gd, () => GlSceneStats.collectInPage({
    mode: '3d', bytesPerVertex: GlSceneStats.BYTES_PER_SCATTER3D_VERTEX,
  }));
  assert.deepEqual(stats, {
    traceCount: 3,
    glObjects: 3,
    pointClouds: 2,
    fillVertices: 900,
    outlineVertices: 400,
    vertexMB: +((1300 * 40) / 1048576).toFixed(1),
    pickStats: { renders: 2, deferredFrames: 5, readbacks: 4, buffers: 2 },
    canvas: { backingWidth: 3200, backingHeight: 1900, cssWidth: 1600, cssHeight: 950, backingRatio: 2 },
  });
});

test('3d stats report null pick stats before the scene exposes them', () => {
  const gd = {
    data: [{}],
    _fullLayout: { scene: { _scene: { glplot: { canvas: canvas(10, 10, 10, 10), objects: [] } } } },
  };
  const stats = withPlot(gd, () => GlSceneStats.collectInPage({ mode: '3d', bytesPerVertex: 40 }));
  assert.equal(stats.pickStats, null);
  assert.equal(stats.fillVertices, 0);
});

test('2d stats read the scattergl context canvas', () => {
  const ctxCanvas = canvas(1600, 950, 1600, 950);
  const gd = {
    data: [{}, {}],
    querySelector: (sel) => (sel === 'canvas.gl-canvas-context' ? ctxCanvas : null),
  };
  const stats = withPlot(gd, () => GlSceneStats.collectInPage({ mode: '2d', bytesPerVertex: 40 }));
  assert.deepEqual(stats, {
    traceCount: 2,
    canvas: { backingWidth: 1600, backingHeight: 950, cssWidth: 1600, cssHeight: 950, backingRatio: 1 },
  });
});

test('the in-page function survives serialization (no module-scope references)', () => {
  // page.evaluate ships only the function source; rebuild it the same way.
  const rebuilt = new Function(`return (${GlSceneStats.collectInPage.toString()})`)();
  const gd = { data: [{}], querySelector: () => null };
  const stats = withPlot(gd, () => rebuilt({ mode: '2d', bytesPerVertex: 40 }));
  assert.deepEqual(stats, { traceCount: 1, canvas: null });
});

test('collect evaluates in the page and reports failures instead of hiding them', async () => {
  const gd = { data: [{}, {}], querySelector: () => null };
  const page = { evaluate: async (fn, args) => withPlot(gd, () => fn(args)) };
  assert.deepEqual(await GlSceneStats.collect(page, '2d'), { traceCount: 2, canvas: null });

  const brokenPage = { evaluate: async () => { throw new Error('Target closed'); } };
  assert.deepEqual(await GlSceneStats.collect(brokenPage, '3d'), { error: 'Error: Target closed' });
});

test('stats are null when the plot is not mounted', () => {
  const stats = withPlot(null, () => GlSceneStats.collectInPage({ mode: '3d', bytesPerVertex: 40 }));
  assert.equal(stats, null);
});

test('summary line names traces, vertices and pick stats', () => {
  const line = GlSceneStats.summarize({
    traceCount: 2695, fillVertices: 1000, outlineVertices: 500, vertexMB: 0.1,
    pickStats: { renders: 3, deferredFrames: 40, readbacks: 33, buffers: 11 },
    canvas: { backingRatio: 2 },
  });
  assert.match(line, /traces 2695/);
  assert.match(line, /outline 500/);
  assert.match(line, /pick buffers 11/);
  assert.match(line, /backing 2x/);
  assert.equal(GlSceneStats.summarize(null), 'gl stats unavailable');
  assert.equal(GlSceneStats.summarize({ error: 'Error: Target closed' }), 'gl stats failed: Error: Target closed');
});
