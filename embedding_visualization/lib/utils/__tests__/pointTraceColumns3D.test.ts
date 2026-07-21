import { describe, expect, it } from 'vitest';

import type { Point3D } from '../../types/types';
import { PointTraceColumns3D } from '../pointTraceColumns3D';

const point = (x: number, y: number, z: number, index: number): Point3D => ({
  x,
  y,
  z,
  index,
  id: `item-${index}`,
  label: `Item ${index}`,
  document: '',
  category: '',
  metadata: {},
});

describe('PointTraceColumns3D', () => {
  it('materializes point coordinates and global indices into compact typed arrays', () => {
    const columns = PointTraceColumns3D.fromPoints([
      point(1.25, -2.5, 3.75, 7),
      point(4.5, 5.25, -6.75, 11),
    ]);

    expect(columns.x).toBeInstanceOf(Float32Array);
    expect(columns.y).toBeInstanceOf(Float32Array);
    expect(columns.z).toBeInstanceOf(Float32Array);
    expect(columns.pointIndices).toBeInstanceOf(Uint32Array);
    expect(Array.from(columns.x)).toEqual([1.25, 4.5]);
    expect(Array.from(columns.y)).toEqual([-2.5, 5.25]);
    expect(Array.from(columns.z)).toEqual([3.75, -6.75]);
    expect(Array.from(columns.pointIndices)).toEqual([7, 11]);
  });

  it('materializes an indexed subset in local trace order while preserving global indices', () => {
    const columns = PointTraceColumns3D.fromIndexedColumns(
      new Float32Array([10, 20, 30]),
      new Float32Array([11, 21, 31]),
      new Float32Array([12, 22, 32]),
      [2, 0],
      new Uint32Array([100, 101, 102]),
    );

    expect(Array.from(columns.x)).toEqual([30, 10]);
    expect(Array.from(columns.y)).toEqual([31, 11]);
    expect(Array.from(columns.z)).toEqual([32, 12]);
    expect(Array.from(columns.pointIndices)).toEqual([102, 100]);
  });

  it('returns zero-length typed arrays for an empty trace', () => {
    const columns = PointTraceColumns3D.fromPoints([]);

    expect(columns.x).toHaveLength(0);
    expect(columns.y).toHaveLength(0);
    expect(columns.z).toHaveLength(0);
    expect(columns.pointIndices).toHaveLength(0);
  });
});
