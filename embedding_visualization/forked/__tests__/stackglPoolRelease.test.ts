/**
 * The stackgl bundle's typedarray-pool keeps every vertex buffer gl-scatter3d hands
 * back to it, on `globalThis.__TYPEDARRAY_POOL`, for the life of the page.
 * `StackglBufferPool` (lib/utils) releases the large ones from app code. These tests
 * run it against the real bundle: they pin the global's name and shape, show the
 * retention it fixes, and check that a point cloud rebuilds identically from an
 * emptied pool. See documentation/RENDER_PERFORMANCE.md, "Typed-array pool release".
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { RecordingGl, StackglHarness, type PointCloud } from './stackglHarness';
import { StackglBufferPool } from '../../lib/utils/stackglBufferPool';

const { gl_scatter3d: createPointCloud } = StackglHarness.load();

const MiB = 1 << 20;
/** Enough points that each of the four vertex buffers exceeds the 1 MiB release threshold. */
const POINT_COUNT = 2000;
const CAMERA = {};

let recorder: RecordingGl;

/** Options as plotly's scatter3d convert.js passes them, for `count` points on a helix. */
function options(count: number): Record<string, unknown> {
  const position: number[][] = [];
  const color: number[][] = [];
  for (let i = 0; i < count; i++) {
    const t = i / count;
    position.push([Math.cos(t * 20), Math.sin(t * 20), t * 4 - 2]);
    color.push([t, 1 - t, 0.5, 1]);
  }
  return {
    gl: recorder.gl,
    position,
    color,
    size: 12,
    glyph: '●',
    opacity: 1,
    orthographic: true,
    lineWidth: 0,
    lineColor: [0, 0, 0, 1],
    project: [false, false, false],
    projectScale: [2 / 3, 2 / 3, 2 / 3],
    projectOpacity: [1, 1, 1],
  };
}

function build(count = POINT_COUNT): PointCloud {
  const cloud = createPointCloud(options(count));
  cloud.axes = { bounds: [[-10, -10, -10], [10, 10, 10]] };
  return cloud;
}

/** The pool bytes an update leaves behind: each of the four uploads rounded up to a power of two. */
function pooledBytesFor(cloud: PointCloud): number {
  return Object.values(recorder.uploadsOf(cloud)).reduce((sum, bytes) => sum + StackglHarness.nextPow2(bytes.length), 0);
}

const bundlePool = () => (globalThis as Record<string, unknown>).__TYPEDARRAY_POOL as { DATA: ArrayBuffer[][] };

// Rasterize the glyph once up front: the bundle borrows scratch buffers for it, and the
// tests below account for every pooled byte.
recorder = new RecordingGl();
build(3);

beforeEach(() => {
  recorder = new RecordingGl();
  StackglBufferPool.release(0);
});

describe('the bundle pool', () => {
  it('is the global StackglBufferPool reads, with 32 byte-size buckets', () => {
    expect(StackglBufferPool.pool()).toBe(bundlePool());
    expect(bundlePool().DATA).toHaveLength(32);
  });

  it('keeps the four vertex buffers of an update, rounded up to powers of two', () => {
    const cloud = build();
    for (const bytes of Object.values(recorder.uploadsOf(cloud))) {
      expect(bytes.length).toBeGreaterThanOrEqual(MiB);
    }
    expect(StackglBufferPool.retainedBytes()).toBe(pooledBytesFor(cloud));
    expect(StackglBufferPool.release()).toEqual({ buffers: 4, bytes: pooledBytesFor(cloud) });
  });

  it('reuses those buffers for a rebuild of the same point count', () => {
    const cloud = build();
    const retained = StackglBufferPool.retainedBytes(0);
    cloud.update(options(POINT_COUNT));
    expect(StackglBufferPool.retainedBytes(0)).toBe(retained);
    expect(StackglBufferPool.release().buffers).toBe(4);
  });

  it('cannot reuse them after the point count changes, so both sets stay resident', () => {
    const cloud = build();
    const large = pooledBytesFor(cloud);
    const base = StackglBufferPool.retainedBytes(0);
    cloud.update(options(POINT_COUNT / 8));
    const small = pooledBytesFor(cloud);
    expect(small).toBeLessThan(large);
    expect(StackglBufferPool.retainedBytes(0)).toBe(base + small);
  });
});

describe('StackglBufferPool.release on the real bundle', () => {
  it('empties the large buckets and leaves small buffers alone', () => {
    build();
    const smallBuffer = new ArrayBuffer(4096);
    bundlePool().DATA[12].push(smallBuffer);

    StackglBufferPool.release();

    expect(StackglBufferPool.retainedBytes()).toBe(0);
    expect(bundlePool().DATA[12]).toContain(smallBuffer);
  });

  it('lets a point cloud rebuild byte-identically from the emptied pool', () => {
    const cloud = build();
    const before = Object.entries(recorder.uploadsOf(cloud)).map(([key, bytes]) => [key, Buffer.from(bytes)] as const);
    StackglBufferPool.release();

    cloud.update(options(POINT_COUNT));
    const after = recorder.uploadsOf(cloud);
    for (const [key, bytes] of before) {
      expect(Buffer.from(after[key as keyof typeof after]).equals(bytes), key).toBe(true);
    }
    expect(StackglBufferPool.retainedBytes()).toBe(pooledBytesFor(cloud));

    recorder.resetCalls();
    cloud.draw(CAMERA);
    expect(recorder.drawCalls.length).toBeGreaterThan(0);
  });
});
