/**
 * Tests for the data->scene model-matrix builder. This matrix is now cached
 * per-bounds in ScatterPlot3D (recomputed only when bounds change), so pinning
 * its exact output guards the cache against silent drift.
 */

import { describe, it, expect } from 'vitest';
import { buildDataToSceneMatrix, type DataBounds } from '../rendeding';

const cube = (min: number, max: number): DataBounds => ({
  minX: min, maxX: max,
  minY: min, maxY: max,
  minZ: min, maxZ: max,
});

describe('buildDataToSceneMatrix', () => {
  it('returns the identity matrix when there is no layout and no fallback bounds', () => {
    const m = buildDataToSceneMatrix(null);
    expect(Array.from(m)).toEqual([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
  });

  it('builds Scale(1/normRange) * Translate(-center) from fallback bounds (cube 0..10)', () => {
    // center = (5,5,5); normRange = cbrt(10*10*10) = 10; s = 0.1
    const m = buildDataToSceneMatrix(null, cube(0, 10));
    // Diagonal scale
    expect(m[0]).toBeCloseTo(0.1);
    expect(m[5]).toBeCloseTo(0.1);
    expect(m[10]).toBeCloseTo(0.1);
    // Translation column (-center * s)
    expect(m[12]).toBeCloseTo(-0.5);
    expect(m[13]).toBeCloseTo(-0.5);
    expect(m[14]).toBeCloseTo(-0.5);
    expect(m[15]).toBe(1);
  });

  it('handles an anisotropic bounds box (per-axis center, uniform scale)', () => {
    // x:[2,6] center 4 range 4 | y:[0,10] center 5 range 10 | z:[-4,4] center 0 range 8
    const bounds: DataBounds = { minX: 2, maxX: 6, minY: 0, maxY: 10, minZ: -4, maxZ: 4 };
    const normRange = Math.cbrt(4 * 10 * 8);
    const s = 1 / normRange;
    const m = buildDataToSceneMatrix(null, bounds);
    expect(m[0]).toBeCloseTo(s);
    expect(m[5]).toBeCloseTo(s);
    expect(m[10]).toBeCloseTo(s);
    expect(m[12]).toBeCloseTo(-4 * s);
    expect(m[13]).toBeCloseTo(-5 * s);
    expect(m[14]).toBeCloseTo(0);
  });

  it('prefers _fullLayout.scene axis ranges over fallback bounds', () => {
    // Ranges [-1,1] on each axis: center 0, normRange = cbrt(8) = 2, s = 0.5.
    const graphDiv = {
      _fullLayout: {
        scene: {
          xaxis: { range: [-1, 1] },
          yaxis: { range: [-1, 1] },
          zaxis: { range: [-1, 1] },
        },
      },
    };
    // Fallback bounds are deliberately different to prove ranges win.
    const m = buildDataToSceneMatrix(graphDiv, cube(0, 100));
    expect(m[0]).toBeCloseTo(0.5);
    expect(m[5]).toBeCloseTo(0.5);
    expect(m[10]).toBeCloseTo(0.5);
    expect(m[12]).toBeCloseTo(0);
    expect(m[13]).toBeCloseTo(0);
    expect(m[14]).toBeCloseTo(0);
  });
});
