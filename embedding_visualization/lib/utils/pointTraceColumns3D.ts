import type { Point3D } from '../types/types';

/** Compact coordinate/index columns passed to Plotly scatter3d traces. */
export class PointTraceColumns3D {
  private constructor(
    readonly x: Float32Array,
    readonly y: Float32Array,
    readonly z: Float32Array,
    readonly pointIndices: Uint32Array,
  ) {}

  static fromPoints(points: readonly Point3D[]): PointTraceColumns3D {
    const x = new Float32Array(points.length);
    const y = new Float32Array(points.length);
    const z = new Float32Array(points.length);
    const pointIndices = new Uint32Array(points.length);

    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      x[i] = point.x;
      y[i] = point.y;
      z[i] = point.z;
      pointIndices[i] = point.index;
    }

    return new PointTraceColumns3D(x, y, z, pointIndices);
  }

  static fromIndexedColumns(
    allX: ArrayLike<number>,
    allY: ArrayLike<number>,
    allZ: ArrayLike<number>,
    indices: readonly number[],
    globalPointIndices: ArrayLike<number>,
  ): PointTraceColumns3D {
    const x = new Float32Array(indices.length);
    const y = new Float32Array(indices.length);
    const z = new Float32Array(indices.length);
    const pointIndices = new Uint32Array(indices.length);

    for (let i = 0; i < indices.length; i++) {
      const sourceIndex = indices[i];
      x[i] = allX[sourceIndex];
      y[i] = allY[sourceIndex];
      z[i] = allZ[sourceIndex];
      pointIndices[i] = globalPointIndices[sourceIndex];
    }

    return new PointTraceColumns3D(x, y, z, pointIndices);
  }
}
