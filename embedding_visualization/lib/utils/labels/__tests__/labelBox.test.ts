import { describe, expect, it } from 'vitest';

import type { BoundingBox } from '../../collisionGrid';
import { LabelBox } from '../labelBox';

function expectBox(actual: BoundingBox, expected: BoundingBox) {
  expect(actual.loX).toBeCloseTo(expected.loX);
  expect(actual.loY).toBeCloseTo(expected.loY);
  expect(actual.hiX).toBeCloseTo(expected.hiX);
  expect(actual.hiY).toBeCloseTo(expected.hiY);
}

describe('LabelBox', () => {
  it('centres a box on its anchor: half the text width and 0.6 of the font size, plus a 4 px pad', () => {
    expectBox(LabelBox.centered(100, 50, 40, 10), { loX: 76, loY: 40, hiX: 124, hiY: 60 });
    // 13 px cluster labels: half-height 0.6 × 13 + 4 = 11.8
    expectBox(LabelBox.centered(0, 0, 20, 13), { loX: -14, loY: -11.8, hiX: 14, hiY: 11.8 });
  });

  it('takes a custom pad', () => {
    expectBox(LabelBox.centered(0, 0, 10, 10, 0), { loX: -5, loY: -6, hiX: 5, hiY: 6 });
  });

  it('boxes a point label starting `offset` px right of its anchor, with 2 px side and 1 px vertical pads', () => {
    // 3D search labels: offset 8, font size 11
    expectBox(LabelBox.rightOf(100, 50, 8, 30, 11), { loX: 106, loY: 42.4, hiX: 140, hiY: 57.6 });
  });

  it('probes a 1 px strip where a point label would start', () => {
    expectBox(LabelBox.probe(100, 50, 8, 11), { loX: 108, loY: 43.4, hiX: 109, hiY: 56.6 });
  });

  it('flags boxes with non-finite coordinates', () => {
    expect(LabelBox.isFinite({ loX: 0, loY: 0, hiX: 1, hiY: 1 })).toBe(true);
    expect(LabelBox.isFinite({ loX: Number.NaN, loY: 0, hiX: 1, hiY: 1 })).toBe(false);
    expect(LabelBox.isFinite({ loX: 0, loY: 0, hiX: Number.POSITIVE_INFINITY, hiY: 1 })).toBe(false);
  });
});
