/**
 * 3D-to-2D projection utilities for the canvas label overlay.
 *
 * Provides MVP matrix construction and screen-space projection from
 * Plotly's glplot camera internals + data bounds.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DataBounds {
  minX: number; maxX: number;
  minY: number; maxY: number;
  minZ: number; maxZ: number;
}

// ---------------------------------------------------------------------------
// Column-major 4x4 matrix math
// ---------------------------------------------------------------------------

/** Multiply two column-major 4x4 matrices: result = A * B */
function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[row]      * b[col * 4]     +
        a[4 + row]  * b[col * 4 + 1] +
        a[8 + row]  * b[col * 4 + 2] +
        a[12 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

/** Multiply a column-major 4x4 matrix by a vec4. */
export function multiplyMat4Vec4(
  m: Float32Array,
  v: [number, number, number, number],
): [number, number, number, number] {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8]  * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9]  * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

/** Compute the combined Model-View-Projection matrix: P * V * M */
export function computeMVP(
  projection: Float32Array,
  view: Float32Array,
  model: Float32Array,
): Float32Array {
  return mat4Multiply(projection, mat4Multiply(view, model));
}

// ---------------------------------------------------------------------------
// Model matrix: data coords -> Plotly scene coords
// ---------------------------------------------------------------------------

/**
 * Build the model matrix that transforms data-space coordinates into the
 * normalized scene coordinates that Plotly's camera operates in.
 *
 * With `aspectmode: 'data'`, Plotly maps:
 *   sceneCoord = (dataCoord - center) / maxRange
 */
export function buildDataToSceneMatrix(bounds: DataBounds): Float32Array {
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  const cz = (bounds.minZ + bounds.maxZ) / 2;
  const maxRange = Math.max(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY,
    bounds.maxZ - bounds.minZ,
  ) || 1;
  const s = 1 / maxRange;

  // Column-major 4x4: Scale(s) * Translate(-center)
  return new Float32Array([
    s, 0, 0, 0,
    0, s, 0, 0,
    0, 0, s, 0,
    -cx * s, -cy * s, -cz * s, 1,
  ]);
}

// ---------------------------------------------------------------------------
// 3D -> 2D projection
// ---------------------------------------------------------------------------

/**
 * Project a 3D point to CSS-pixel screen coordinates via an MVP matrix.
 * Returns null if the point is behind the camera (w <= 0).
 */
export function projectToScreen(
  x: number,
  y: number,
  z: number,
  mvp: Float32Array,
  viewportWidth: number,
  viewportHeight: number,
): { x: number; y: number } | null {
  const clip = multiplyMat4Vec4(mvp, [x, y, z, 1]);
  const w = clip[3];
  if (w <= 0) return null;

  const ndcX = clip[0] / w;
  const ndcY = clip[1] / w;

  return {
    x: (ndcX + 1) * 0.5 * viewportWidth,
    y: (1 - ndcY) * 0.5 * viewportHeight, // Y flipped
  };
}
