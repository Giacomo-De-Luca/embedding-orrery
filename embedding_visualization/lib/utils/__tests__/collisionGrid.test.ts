import { describe, expect, it } from 'vitest';

import { CollisionGrid } from '../collisionGrid';

/** A 100 × 100 px grid of 10 px cells. */
const grid = () => new CollisionGrid({ loX: 0, loY: 0, hiX: 100, hiY: 100 }, 10, 10);

describe('CollisionGrid', () => {
  it('accepts a free box and rejects one that overlaps it', () => {
    const g = grid();
    expect(g.insert({ loX: 10, loY: 10, hiX: 30, hiY: 20 })).toBe(true);
    expect(g.insert({ loX: 25, loY: 15, hiX: 40, hiY: 25 })).toBe(false);
    expect(g.insert({ loX: 31, loY: 10, hiX: 40, hiY: 20 })).toBe(true);
  });

  it('only tests a box when asked to, without inserting it', () => {
    const g = grid();
    expect(g.insert({ loX: 10, loY: 10, hiX: 20, hiY: 20 }, true)).toBe(true);
    expect(g.insert({ loX: 10, loY: 10, hiX: 20, hiY: 20 })).toBe(true);
    expect(g.insert({ loX: 15, loY: 15, hiX: 16, hiY: 16 }, true)).toBe(false);
  });

  it('counts touching edges as a collision', () => {
    const g = grid();
    g.insert({ loX: 0, loY: 0, hiX: 10, hiY: 10 });
    expect(g.insert({ loX: 10, loY: 0, hiX: 20, hiY: 10 })).toBe(false);
  });

  it('rejects boxes entirely outside the bounds and accepts boxes partly inside', () => {
    const g = grid();
    expect(g.insert({ loX: 101, loY: 0, hiX: 120, hiY: 10 })).toBe(false);
    expect(g.insert({ loX: -20, loY: -20, hiX: -1, hiY: -1 })).toBe(false);
    expect(g.insert({ loX: 90, loY: 90, hiX: 130, hiY: 130 })).toBe(true);
    expect(g.insert({ loX: 95, loY: 95, hiX: 99, hiY: 99 })).toBe(false);
  });

  it('finds collisions with a box spanning many cells, in any of its cells', () => {
    const g = grid();
    g.insert({ loX: 5, loY: 5, hiX: 95, hiY: 95 });
    expect(g.insert({ loX: 50, loY: 50, hiX: 52, hiY: 52 })).toBe(false);
    expect(g.insert({ loX: 90, loY: 8, hiX: 92, hiY: 9 })).toBe(false);
    expect(g.insert({ loX: 96, loY: 96, hiX: 99, hiY: 99 })).toBe(true);
  });

  it('rejects boxes with non-finite coordinates, which then block nothing', () => {
    const g = grid();
    expect(g.insert({ loX: Number.NaN, loY: 10, hiX: 20, hiY: 20 })).toBe(false);
    expect(g.insert({ loX: 10, loY: 10, hiX: Number.POSITIVE_INFINITY, hiY: 20 })).toBe(false);
    expect(g.insert({ loX: 10, loY: 10, hiX: 20, hiY: 20 })).toBe(true);
  });

  it('covers a viewport with 25 columns and 50 rows of cells, at least 1 px each', () => {
    expect(CollisionGrid.forViewport(500, 400))
      .toEqual(new CollisionGrid({ loX: 0, loY: 0, hiX: 500, hiY: 400 }, 20, 8));
    expect(CollisionGrid.forViewport(10, 10))
      .toEqual(new CollisionGrid({ loX: 0, loY: 0, hiX: 10, hiY: 10 }, 1, 1));
  });
});
