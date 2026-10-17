/**
 * WebGL scene statistics read from the live plot.
 *
 * `collectInPage` runs inside the page via Playwright's `page.evaluate`, which
 * serializes the function source: it must stay self-contained (no references
 * to module scope) and return plain JSON.
 */
export class GlSceneStats {
  /** Bytes per gl-scatter3d vertex: position 3×f32 + colour 4×f32 + glyph 2×f32 + id 4×u8. */
  static BYTES_PER_SCATTER3D_VERTEX = 40;

  static collectInPage = function collectGlSceneStats({ mode, bytesPerVertex }) {
    const gd = document.querySelector('.js-plotly-plot');
    if (!gd || !gd.data) return null;
    const canvasStats = (canvas) => (canvas && canvas.clientWidth ? {
      backingWidth: canvas.width,
      backingHeight: canvas.height,
      cssWidth: canvas.clientWidth,
      cssHeight: canvas.clientHeight,
      backingRatio: +(canvas.width / canvas.clientWidth).toFixed(2),
    } : null);

    const stats = { traceCount: gd.data.length };
    // Idle ArrayBuffers in the stackgl typed-array pool (gl-scatter3d vertex arrays
    // after upload). lib/utils/stackglBufferPool.ts releases the large ones.
    const pool = globalThis.__TYPEDARRAY_POOL;
    stats.poolRetainedMB = pool && Array.isArray(pool.DATA)
      ? +(pool.DATA.flat().reduce((n, b) => n + b.byteLength, 0) / 1048576).toFixed(1)
      : null;
    if (mode !== '3d') {
      stats.canvas = canvasStats(gd.querySelector('canvas.gl-canvas-context'));
      return stats;
    }
    const glplot = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene
      && gd._fullLayout.scene._scene.glplot;
    if (!glplot) return stats;
    // gl-scatter3d point clouds are the objects carrying a glyph-outline count.
    const clouds = glplot.objects.filter((o) => typeof o.lineVertexCount === 'number');
    const fillVertices = clouds.reduce((n, o) => n + (o.vertexCount || 0), 0);
    const outlineVertices = clouds.reduce((n, o) => n + o.lineVertexCount, 0);
    stats.glObjects = glplot.objects.length;
    stats.pointClouds = clouds.length;
    stats.fillVertices = fillVertices;
    stats.outlineVertices = outlineVertices;
    stats.vertexMB = +(((fillVertices + outlineVertices) * bytesPerVertex) / 1048576).toFixed(1);
    stats.pickStats = glplot.pickStats ? { ...glplot.pickStats } : null;
    stats.canvas = canvasStats(glplot.canvas);
    return stats;
  };

  /** Reads the stats from a Playwright page; a failed read is reported, not swallowed. */
  static async collect(page, mode) {
    try {
      return await page.evaluate(GlSceneStats.collectInPage, {
        mode,
        bytesPerVertex: GlSceneStats.BYTES_PER_SCATTER3D_VERTEX,
      });
    } catch (e) {
      return { error: String(e) };
    }
  }

  /** One console line for the harness log. */
  static summarize(stats) {
    if (!stats) return 'gl stats unavailable';
    if (stats.error) return `gl stats failed: ${stats.error}`;
    const parts = [`traces ${stats.traceCount}`];
    if (stats.fillVertices !== undefined) {
      parts.push(`vertices fill ${stats.fillVertices} / outline ${stats.outlineVertices} (${stats.vertexMB} MB)`);
    }
    if (stats.pickStats) {
      const p = stats.pickStats;
      parts.push(`pick buffers ${p.buffers} | renders ${p.renders} | deferred ${p.deferredFrames} | readbacks ${p.readbacks}`);
    }
    if (stats.canvas) parts.push(`backing ${stats.canvas.backingRatio}x`);
    if (typeof stats.poolRetainedMB === 'number') parts.push(`pool ${stats.poolRetainedMB} MB`);
    return parts.join(' | ');
  }
}
