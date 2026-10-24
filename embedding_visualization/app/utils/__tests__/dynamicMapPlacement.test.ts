import { describe, expect, it } from 'vitest';

import { dynamicLabelPlacement, type Label } from '../dynamicMapPlacement';
import { referenceDynamicLabelPlacement } from './fixtures/referenceDynamicMapPlacement';

/** Deterministic PRNG, so every run checks the same layouts. */
function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

type Spread = 'clustered' | 'uniform' | 'packed';

interface LayoutOptions {
  /** Every label gets its own minScale / maxScale. */
  ownRanges?: boolean;
  /** Boxes shifted off their anchors (the anchor can lie outside the box). */
  offsetBoxes?: boolean;
}

/**
 * Labels as the 2D plot builds them: anchors in an 800 × 600 plot, boxes of
 * monospace text widths around them. Priorities repeat (ties), and some labels
 * share an anchor.
 */
function layout(n: number, seed: number, spread: Spread, options: LayoutOptions = {}): Label[] {
  const rand = rng(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const blobs = Array.from({ length: Math.max(2, Math.round(n / 25)) }, () => ({ x: 60 + rand() * 680, y: 40 + rand() * 520, r: 10 + rand() * 50 }));
  const labels: Label[] = [];
  for (let i = 0; i < n; i++) {
    let x: number;
    let y: number;
    if (i > 0 && rand() < 0.05) {
      ({ x, y } = labels[Math.floor(rand() * i)].locationAtZero);
    } else if (spread === 'uniform') {
      x = rand() * 800;
      y = rand() * 600;
    } else if (spread === 'packed') {
      x = 380 + rand() * 40;
      y = 280 + rand() * 40;
    } else {
      const blob = blobs[Math.floor(rand() * blobs.length)];
      x = blob.x + gauss() * blob.r;
      y = blob.y + gauss() * blob.r;
    }
    const halfWidth = ((4 + Math.floor(rand() ** 2 * 30)) * 7.8) / 2 + 4;
    const halfHeight = (rand() < 0.5 ? 12 : 14) * 0.6 + 4;
    const shiftX = options.offsetBoxes ? (rand() - 0.5) * 3 * halfWidth : 0;
    const shiftY = options.offsetBoxes ? (rand() - 0.5) * 3 * halfHeight : 0;
    const label: Label = {
      bounds: { xMin: x - halfWidth + shiftX, yMin: y - halfHeight + shiftY, xMax: x + halfWidth + shiftX, yMax: y + halfHeight + shiftY },
      locationAtZero: { x, y },
      priority: Math.floor(rand() * 8),
    };
    if (options.ownRanges && rand() < 0.6) {
      label.minScale = rand() < 0.5 ? 0 : rand() * 0.8;
      label.maxScale = 0.5 + rand() * 2;
    }
    labels.push(label);
  }
  return labels;
}

/** The fast placement must equal the original, label for label. */
function expectSameAsOriginal(labels: Label[], globalMaxScale?: number) {
  const options = globalMaxScale === undefined ? {} : { globalMaxScale };
  expect(dynamicLabelPlacement(labels, options)).toEqual(referenceDynamicLabelPlacement(labels, options));
}

describe('dynamicLabelPlacement', () => {
  it('places labels exactly as the original on random layouts', () => {
    for (const spread of ['clustered', 'uniform', 'packed'] as const) {
      for (const n of [0, 1, 2, 7, 40, 150, 400]) {
        for (const seed of [1, 2, 3]) {
          const labels = layout(n, seed * 31 + n, spread);
          expectSameAsOriginal(labels, 2);
          expectSameAsOriginal(labels, 1);
        }
      }
    }
  });

  it('matches it with the default global max scale', () => {
    expectSameAsOriginal(layout(120, 5, 'clustered'));
  });

  it('matches it when labels have their own scale ranges, so some leave their range mid-sweep', () => {
    for (const seed of [11, 12, 13, 14]) {
      expectSameAsOriginal(layout(200, seed, 'clustered', { ownRanges: true }), 2);
      expectSameAsOriginal(layout(60, seed, 'packed', { ownRanges: true }), 2);
    }
  });

  it('matches it when boxes do not contain their anchors', () => {
    for (const seed of [21, 22]) {
      expectSameAsOriginal(layout(150, seed, 'clustered', { offsetBoxes: true }), 2);
    }
  });

  it('matches it on many small, crowded layouts', () => {
    for (let seed = 100; seed < 300; seed++) {
      expectSameAsOriginal(layout(5 + (seed % 20), seed, 'packed', { ownRanges: seed % 3 === 0 }), 2);
    }
  });

  it('shows far-apart labels over the whole range and hides the weaker of two that overlap', () => {
    const box = (x: number, y: number, priority: number): Label => ({
      bounds: { xMin: x - 20, yMin: y - 10, xMax: x + 20, yMax: y + 10 },
      locationAtZero: { x, y },
      priority,
    });
    const [left, right] = dynamicLabelPlacement([box(100, 100, 1), box(700, 500, 1)], { globalMaxScale: 2 });
    expect(left).toEqual({ minScale: 0, maxScale: 2 });
    expect(right).toEqual({ minScale: 0, maxScale: 2 });

    const [strong, weak] = dynamicLabelPlacement([box(100, 100, 9), box(110, 100, 1)], { globalMaxScale: 2 });
    expect(strong).toEqual({ minScale: 0, maxScale: 2 });
    expect(weak!.maxScale).toBeLessThan(1);
  });
});
