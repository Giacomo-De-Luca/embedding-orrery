/**
 * Tests for the density-based cluster-label pipeline (pure parts only — the
 * WASM findClusters call is a thin seam kept out of unit tests).
 */

import { describe, it, expect } from 'vitest';
import {
  clustersFromFindClustersResult,
  resolveClusterLabelTexts,
  computeDensityLabelPlacements,
  type DensityCluster,
  type LabeledDensityCluster,
} from '../densityClusterLabels';
import type { Cluster } from '../../../lib/density-clustering';
import type { DensityMapResult } from '../densityUtils';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Density-map bounds: 100×100 grid over data x∈[0,10), y∈[20,40). */
const mapResult: DensityMapResult = {
  density: new Float32Array(0), // unused by the post-processing
  width: 100,
  height: 100,
  xMin: 0,
  xMax: 10,
  yMin: 20,
  yMax: 40,
};

function rawCluster(overrides: Partial<Cluster>): Cluster {
  return {
    identifier: 1,
    sumDensity: 100,
    meanX: 50,
    meanY: 50,
    maxDensity: 10,
    maxDensityLocation: [50, 50],
    pixelCount: 25,
    boundary: [[[45, 45], [55, 45], [55, 55], [45, 55]]],
    boundaryRectApproximation: [[45, 45, 55, 55]],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// clustersFromFindClustersResult
// ---------------------------------------------------------------------------

describe('clustersFromFindClustersResult', () => {
  it('converts pixel coordinates to data space via the map bounds', () => {
    const [c] = clustersFromFindClustersResult([rawCluster({})], mapResult, 0, 0.005);
    // gx=50 on a 100-wide grid over [0,10): x = 0 + 50·(10−0)/99
    expect(c.x).toBeCloseTo((50 * 10) / 99, 10);
    expect(c.y).toBeCloseTo(20 + (50 * 20) / 99, 10);
    expect(c.rects).toHaveLength(1);
    expect(c.rects[0].xMin).toBeCloseTo((45 * 10) / 99, 10);
    expect(c.rects[0].xMax).toBeCloseTo((55 * 10) / 99, 10);
    expect(c.rects[0].yMin).toBeCloseTo(20 + (45 * 20) / 99, 10);
    expect(c.level).toBe(0);
  });

  it('filters clusters below the relative density threshold', () => {
    const clusters = clustersFromFindClustersResult(
      [
        rawCluster({ identifier: 1, sumDensity: 1000 }),
        rawCluster({ identifier: 2, sumDensity: 3 }), // 0.003 of max → dropped
        rawCluster({ identifier: 3, sumDensity: 100 }),
      ],
      mapResult,
      1,
      0.005,
    );
    expect(clusters.map((c) => c.sumDensity)).toEqual([1000, 100]);
    expect(clusters.every((c) => c.level === 1)).toBe(true);
  });

  it('drops clusters without a rect approximation', () => {
    const clusters = clustersFromFindClustersResult(
      [rawCluster({ boundaryRectApproximation: undefined })],
      mapResult,
      0,
      0.005,
    );
    expect(clusters).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// resolveClusterLabelTexts
// ---------------------------------------------------------------------------

const clusterAt = (
  xMin: number, xMax: number, yMin: number, yMax: number,
  level: 0 | 1 = 0, sumDensity = 100,
): DensityCluster => ({
  x: (xMin + xMax) / 2,
  y: (yMin + yMax) / 2,
  sumDensity,
  rects: [{ xMin, xMax, yMin, yMax }],
  level,
});

describe('resolveClusterLabelTexts', () => {
  it('assigns the majority topic label of member points', () => {
    const clusters = [clusterAt(0, 10, 0, 10), clusterAt(20, 30, 0, 10)];
    const points = [
      // Cluster 0: 2× "Poetry", 1× "Prose"
      { x: 1, y: 1, text: 'a' }, { x: 2, y: 2, text: 'b' }, { x: 3, y: 3, text: 'c' },
      // Cluster 1: all "Chemistry"
      { x: 25, y: 5, text: 'd' }, { x: 26, y: 5, text: 'e' },
      // Outside both
      { x: 100, y: 100, text: 'f' },
    ];
    const topics = ['Poetry', 'Poetry', 'Prose', 'Chemistry', 'Chemistry', 'Physics'];
    const result = resolveClusterLabelTexts(clusters, points, (i) => topics[i]);
    expect(result[0].text).toBe('Poetry');
    expect(result[0].topicLabel).toBe('Poetry');
    expect(result[1].text).toBe('Chemistry');
  });

  it('breaks topic-count ties lexicographically (deterministic)', () => {
    const clusters = [clusterAt(0, 10, 0, 10)];
    const points = [
      { x: 1, y: 1, text: 'a' }, { x: 2, y: 2, text: 'b' },
      { x: 3, y: 3, text: 'c' }, { x: 4, y: 4, text: 'd' },
    ];
    const topics = ['Zebra', 'Apple', 'Zebra', 'Apple'];
    const result = resolveClusterLabelTexts(clusters, points, (i) => topics[i]);
    expect(result[0].text).toBe('Apple');
  });

  // The summarizer's c-TF-IDF drops words with document frequency < 2 (word
  // counts are 1/wordCount-weighted), so fixtures need repeated vocabulary.
  const repeatDocs = (
    xBase: number, phrases: string[], count: number,
  ): { x: number; y: number; text: string }[] =>
    Array.from({ length: count }, (_, i) => ({
      x: xBase + (i % 10) * 0.5,
      y: 1 + (i % 5),
      text: phrases[i % phrases.length],
    }));

  it('falls back to the summarizer for clusters with no topic members', () => {
    const clusters = [clusterAt(0, 10, 0, 10), clusterAt(20, 30, 0, 10)];
    const quantumDocs = repeatDocs(0, ['quantum entanglement research', 'quantum computing research'], 12);
    const poetryDocs = repeatDocs(20, ['medieval poetry analysis', 'medieval poetry manuscripts'], 12);
    const points = [...quantumDocs, ...poetryDocs];
    // Topics only for cluster 1's points; cluster 0 members have none
    const result = resolveClusterLabelTexts(clusters, points, (i) => (i >= quantumDocs.length ? 'Poetry' : null));
    expect(result[1].text).toBe('Poetry');
    expect(result[0].topicLabel).toBeNull();
    expect(result[0].text.length).toBeGreaterThan(0);
    expect(result[0].text).toMatch(/quantum|research|computing|entanglement/i);
  });

  it('uses the summarizer for every cluster when no topic accessor is given', () => {
    const clusters = [clusterAt(0, 10, 0, 10), clusterAt(20, 30, 0, 10)];
    const points = [
      ...repeatDocs(0, ['neural network training', 'neural network inference'], 12),
      ...repeatDocs(20, ['baroque violin concerto', 'baroque cello concerto'], 12),
    ];
    const result = resolveClusterLabelTexts(clusters, points, null);
    expect(result[0].topicLabel).toBeNull();
    expect(result[1].topicLabel).toBeNull();
    expect(result[0].text).toMatch(/neural|network/i);
    expect(result[1].text).toMatch(/baroque|concerto/i);
  });
});

// ---------------------------------------------------------------------------
// computeDensityLabelPlacements — per-level scale bands
// ---------------------------------------------------------------------------

/** Minimal measureText stub — node has no canvas. */
const fakeCtx = {
  font: '',
  measureText: (text: string) => ({ width: text.length * 7 }),
} as unknown as CanvasRenderingContext2D;

const ranges = { xRange: [0, 100] as [number, number], yRange: [0, 100] as [number, number] };
const plotArea = { left: 0, top: 0, width: 800, height: 600 };

const labeled = (
  overrides: Partial<LabeledDensityCluster> & { x: number; y: number },
): LabeledDensityCluster => ({
  sumDensity: 100,
  rects: [],
  level: 0,
  text: 'Label',
  topicLabel: null,
  ...overrides,
});

describe('computeDensityLabelPlacements', () => {
  it('assigns Apple’s per-level scale bands when both levels are present', () => {
    const placements = computeDensityLabelPlacements(
      [
        labeled({ x: 20, y: 20, level: 0, text: 'Coarse' }),
        labeled({ x: 80, y: 80, level: 1, text: 'Fine' }),
      ],
      fakeCtx, ranges, plotArea, () => '#888888',
    );
    const coarse = placements.find((p) => p.level === 0)!;
    const fine = placements.find((p) => p.level === 1)!;
    // Far apart → no conflicts → placements exist and respect the level bands
    expect(coarse.placement).not.toBeNull();
    expect(fine.placement).not.toBeNull();
    // Level 0: minScale = 1/(2·2⁰·1.2) ≈ 0.4167, visible out to the global max
    expect(coarse.placement!.minScale).toBeCloseTo(1 / 2.4, 6);
    expect(coarse.placement!.maxScale).toBeGreaterThanOrEqual(2);
    // Level 1: maxScale = 1/(2·2⁰) = 0.5, visible down to full zoom-in
    expect(fine.placement!.maxScale).toBeCloseTo(0.5, 6);
    expect(fine.placement!.minScale).toBeLessThanOrEqual(1e-6);
  });

  it('single-level input gets no per-level band constraints', () => {
    const placements = computeDensityLabelPlacements(
      [labeled({ x: 20, y: 20, level: 0 }), labeled({ x: 80, y: 80, level: 0, text: 'Other' })],
      fakeCtx, ranges, plotArea, () => '#888888',
    );
    for (const p of placements) {
      expect(p.placement).not.toBeNull();
      expect(p.placement!.minScale).toBeLessThanOrEqual(1e-6);
      expect(p.placement!.maxScale).toBeGreaterThanOrEqual(2);
    }
  });

  it('sets per-level font sizes and carries text/color/priority through', () => {
    const placements = computeDensityLabelPlacements(
      [
        labeled({ x: 20, y: 20, level: 0, sumDensity: 500, topicLabel: 'Poetry', text: 'Poetry' }),
        labeled({ x: 80, y: 80, level: 1, sumDensity: 50, text: 'fine words' }),
      ],
      fakeCtx, ranges, plotArea, (topic) => (topic === 'Poetry' ? '#ff0000' : '#888888'),
    );
    const coarse = placements.find((p) => p.level === 0)!;
    const fine = placements.find((p) => p.level === 1)!;
    expect(coarse.fontSize).toBe(14);
    expect(fine.fontSize).toBe(12);
    expect(coarse.priority).toBe(500);
    expect(coarse.color).toBe('#ff0000');
    expect(fine.color).toBe('#888888');
    expect(coarse.label).toBe('Poetry');
    expect(coarse.topicLabel).toBe('Poetry');
  });
});
