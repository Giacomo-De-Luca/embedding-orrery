/**
 * Releases idle buffers from the stackgl typed-array pool.
 *
 * gl-scatter3d (in the forked Plotly's prebuilt `stackgl_modules/index.js`) borrows
 * its four vertex arrays from `typedarray-pool`, uploads them to the GPU and hands
 * them straight back. The pool files every returned ArrayBuffer in a power-of-two
 * bucket on `globalThis.__TYPEDARRAY_POOL.DATA` and never empties it (its
 * `clearCache` is unreachable from the bundle's exports). After a large collection
 * has been drawn in 3D its buffers therefore stay resident for the life of the tab
 * (hundreds of MiB; the WordNet figures are in the doc below) through collection
 * switches, 2D mode and client-side navigation, because smaller collections borrow
 * from other buckets.
 *
 * Buffers in use are never in the buckets, so releasing is safe at any moment: the
 * next `update()` allocates afresh. Call it when the pooled sizes can no longer be
 * reused (the point count changed, the 3D plot unmounted), not after every render,
 * or each rebuild would pay for fresh allocations instead of reusing the pool.
 * See documentation/RENDER_PERFORMANCE.md, "Typed-array pool release".
 */

/** What one release call dropped. */
export interface PoolRelease {
  buffers: number;
  bytes: number;
}

/** The part of typedarray-pool's global this class touches. */
interface TypedArrayPool {
  /** `DATA[i]` holds idle ArrayBuffers whose byte length has floor(log2) = i. */
  DATA: ArrayBuffer[][];
}

export class StackglBufferPool {
  /** typedarray-pool shares its cache across copies through this global. */
  static readonly GLOBAL_KEY = '__TYPEDARRAY_POOL';
  /** Buffers at or above 1 MiB (bucket 20) are released by default. */
  static readonly DEFAULT_MIN_BYTES = 1 << 20;

  /** The bundle's pool, or null before the bundle has loaded (and on the server). */
  static pool(): TypedArrayPool | null {
    const candidate = (globalThis as Record<string, unknown>)[StackglBufferPool.GLOBAL_KEY];
    if (!candidate || typeof candidate !== 'object') return null;
    return Array.isArray((candidate as { DATA?: unknown }).DATA) ? (candidate as TypedArrayPool) : null;
  }

  /** Bytes of idle buffers at or above `minBytes`. */
  static retainedBytes(minBytes = StackglBufferPool.DEFAULT_MIN_BYTES): number {
    let bytes = 0;
    StackglBufferPool.forEachBucket((bucket) => {
      for (const buffer of bucket) if (buffer.byteLength >= minBytes) bytes += buffer.byteLength;
    });
    return bytes;
  }

  /** Drops idle buffers at or above `minBytes` so the garbage collector can reclaim them. */
  static release(minBytes = StackglBufferPool.DEFAULT_MIN_BYTES): PoolRelease {
    const dropped: PoolRelease = { buffers: 0, bytes: 0 };
    StackglBufferPool.forEachBucket((bucket) => {
      const kept = bucket.filter((buffer) => {
        if (buffer.byteLength < minBytes) return true;
        dropped.buffers += 1;
        dropped.bytes += buffer.byteLength;
        return false;
      });
      if (kept.length === bucket.length) return;
      // In place. The bundle captured `DATA` itself and indexes into it on every
      // malloc and free, so that table must survive; emptying its buckets in place
      // is the most conservative way to keep it intact.
      bucket.length = 0;
      bucket.push(...kept);
    });
    return dropped;
  }

  private static forEachBucket(visit: (bucket: ArrayBuffer[]) => void): void {
    const pool = StackglBufferPool.pool();
    if (!pool) return;
    for (const bucket of pool.DATA) {
      if (Array.isArray(bucket) && bucket.length > 0) visit(bucket);
    }
  }
}
