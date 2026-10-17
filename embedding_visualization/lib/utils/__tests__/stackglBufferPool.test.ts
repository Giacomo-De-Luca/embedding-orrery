/**
 * StackglBufferPool releases idle buffers from typedarray-pool's cache, which the
 * forked Plotly's stackgl bundle keeps on `globalThis.__TYPEDARRAY_POOL`. These tests
 * use a pool shaped like the real one; forked/__tests__/stackglPoolRelease.test.ts
 * runs the same class against the real bundle.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { StackglBufferPool } from '../stackglBufferPool';

const KiB = 1024;
const MiB = 1024 * KiB;

/** Buckets like typedarray-pool's: `DATA[i]` holds ArrayBuffers whose floor(log2(byteLength)) is i. */
function fakePool(byteLengths: number[]) {
  const DATA: ArrayBuffer[][] = Array.from({ length: 32 }, () => []);
  for (const n of byteLengths) DATA[Math.floor(Math.log2(n))].push(new ArrayBuffer(n));
  return { DATA, UINT8: Array.from({ length: 32 }, () => []) };
}

function install(pool: unknown) {
  vi.stubGlobal(StackglBufferPool.GLOBAL_KEY, pool);
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('StackglBufferPool without the bundle', () => {
  it('is a no-op when the pool global is absent', () => {
    expect(StackglBufferPool.pool()).toBeNull();
    expect(StackglBufferPool.retainedBytes()).toBe(0);
    expect(StackglBufferPool.release()).toEqual({ buffers: 0, bytes: 0 });
  });

  it('tolerates a pool without DATA buckets', () => {
    install({ UINT8: [] });
    expect(StackglBufferPool.pool()).toBeNull();
    expect(StackglBufferPool.release()).toEqual({ buffers: 0, bytes: 0 });
  });
});

describe('StackglBufferPool.release', () => {
  it('drops buffers at or above 1 MiB and keeps smaller ones', () => {
    const pool = fakePool([256 * KiB, MiB, 2 * MiB, 8 * MiB]);
    install(pool);
    expect(StackglBufferPool.DEFAULT_MIN_BYTES).toBe(MiB);
    expect(StackglBufferPool.retainedBytes(0)).toBe(256 * KiB + 11 * MiB);
    expect(StackglBufferPool.retainedBytes()).toBe(11 * MiB);

    expect(StackglBufferPool.release()).toEqual({ buffers: 3, bytes: 11 * MiB });

    expect(StackglBufferPool.retainedBytes(0)).toBe(256 * KiB);
    expect(StackglBufferPool.retainedBytes()).toBe(0);
    expect(pool.DATA[18]).toHaveLength(1);
    expect(pool.DATA[20]).toHaveLength(0);
    expect(pool.DATA[23]).toHaveLength(0);
  });

  it('takes a custom inclusive threshold', () => {
    const pool = fakePool([MiB, 2 * MiB, 8 * MiB]);
    install(pool);
    expect(StackglBufferPool.release(2 * MiB)).toEqual({ buffers: 2, bytes: 10 * MiB });
    expect(pool.DATA[20]).toHaveLength(1);
    expect(StackglBufferPool.retainedBytes(2 * MiB)).toBe(0);
  });

  it('releases everything at threshold 0', () => {
    install(fakePool([4 * KiB, MiB]));
    expect(StackglBufferPool.release(0)).toEqual({ buffers: 2, bytes: 4 * KiB + MiB });
    expect(StackglBufferPool.retainedBytes(0)).toBe(0);
  });

  it('measures real byte lengths, not bucket sizes', () => {
    // A 1.5 MiB buffer lands in bucket 20 (floor of log2); its own size is what counts.
    install(fakePool([1.5 * MiB]));
    expect(StackglBufferPool.retainedBytes()).toBe(1.5 * MiB);
    expect(StackglBufferPool.release()).toEqual({ buffers: 1, bytes: 1.5 * MiB });
  });

  it('empties buckets in place, since the bundle holds references to them', () => {
    const pool = fakePool([MiB, MiB, 4 * MiB]);
    install(pool);
    const data = pool.DATA;
    const bucket20 = pool.DATA[20];
    StackglBufferPool.release();
    expect(StackglBufferPool.pool()?.DATA).toBe(data);
    expect(pool.DATA[20]).toBe(bucket20);
    expect(bucket20).toHaveLength(0);
  });

  it('is idempotent', () => {
    install(fakePool([MiB, 4 * MiB]));
    StackglBufferPool.release();
    expect(StackglBufferPool.release()).toEqual({ buffers: 0, bytes: 0 });
  });
});
