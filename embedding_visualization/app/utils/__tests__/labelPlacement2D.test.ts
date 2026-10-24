import { describe, expect, it } from 'vitest';

import { CartesianView } from '../../../lib/utils/cartesianView';
import type { ClusterData } from '../../../lib/utils/clusterGeometry';
import { LabelFonts, type TextMeasurer } from '../../../lib/utils/labels/labelStyle';
import { ZoomBandPlacement, type ClusterLabelPlacement } from '../labelPlacement2D';
import type { Placement } from '../dynamicMapPlacement';

/** Text measuring without a canvas: 7 px per character, recording each font it is set to. */
function measurer() {
  const fonts: string[] = [];
  const ctx: TextMeasurer = {
    get font() { return fonts[fonts.length - 1] ?? ''; },
    set font(font: string) { fonts.push(font); },
    measureText: (text: string) => ({ width: text.length * 7 }),
  };
  return { ctx, fonts };
}

const initial = new CartesianView(
  { xRange: [0, 100], yRange: [0, 100] },
  { left: 0, top: 0, width: 800, height: 600 },
);

describe('ZoomBandPlacement', () => {
  it('lays a label out as a centred box at the initial view', () => {
    const label = ZoomBandPlacement.labelFor({ label: 'x', dataX: 50, dataY: 50, priority: 3, fontSize: 13 }, 70, initial);
    expect(label.locationAtZero).toEqual({ x: 400, y: 300 });
    expect(label.bounds.xMin).toBeCloseTo(361);
    expect(label.bounds.xMax).toBeCloseTo(439);
    // Half-height 0.6 × 13 + 4 = 11.8 (centroid labels used a hard-coded 12)
    expect(label.bounds.yMin).toBeCloseTo(288.2);
    expect(label.bounds.yMax).toBeCloseTo(311.8);
    expect(label.priority).toBe(3);
  });

  it('measures each entry at its own font size and keeps its other fields', () => {
    const { ctx, fonts } = measurer();
    const [a, b] = ZoomBandPlacement.place([
      { label: 'Alpha', dataX: 10, dataY: 10, priority: 5, fontSize: 14, tag: 'a' },
      { label: 'Be', dataX: 90, dataY: 90, priority: 1, fontSize: 12, tag: 'b' },
    ], ctx, initial);
    expect(fonts).toEqual([LabelFonts.cluster(14), LabelFonts.cluster(12)]);
    expect(a.textWidth).toBe(35);
    expect(a.tag).toBe('a');
    expect(b.textWidth).toBe(14);
    expect(b.tag).toBe('b');
  });

  it('shows labels that never overlap across the whole zoom range', () => {
    const placed = ZoomBandPlacement.place([
      { label: 'Left', dataX: 10, dataY: 10, priority: 1, fontSize: 13 },
      { label: 'Right', dataX: 90, dataY: 90, priority: 1, fontSize: 13 },
    ], measurer().ctx, initial);
    for (const { placement } of placed) {
      expect(placement).not.toBeNull();
      expect(placement!.minScale).toBeLessThanOrEqual(1e-6);
      expect(placement!.maxScale).toBeGreaterThanOrEqual(ZoomBandPlacement.GLOBAL_MAX_SCALE);
    }
  });

  it('hides the lower-priority label of two that overlap in the full view', () => {
    const [high, low] = ZoomBandPlacement.place([
      { label: 'High', dataX: 50, dataY: 50, priority: 10, fontSize: 13 },
      { label: 'Low', dataX: 51, dataY: 50, priority: 1, fontSize: 13 },
    ], measurer().ctx, initial);
    expect(high.placement!.maxScale).toBeGreaterThanOrEqual(ZoomBandPlacement.GLOBAL_MAX_SCALE);
    expect(low.placement === null || low.placement.maxScale < 1).toBe(true);
  });

  it('returns the placements highest priority first, ties in their order', () => {
    const placed = ZoomBandPlacement.place([
      { label: 'low', dataX: 10, dataY: 10, priority: 1, fontSize: 13 },
      { label: 'high', dataX: 50, dataY: 50, priority: 9, fontSize: 13 },
      { label: 'also low', dataX: 90, dataY: 90, priority: 1, fontSize: 13 },
    ], measurer().ctx, initial);
    expect(placed.map(p => p.label)).toEqual(['high', 'low', 'also low']);
  });
});

describe('ZoomBandPlacement.forClusters', () => {
  it('places one 13 px label per cluster at its centroid, by point count', () => {
    const clusters = new Map<string, ClusterData>([
      ['Physics', { centroid: { x: 90, y: 90, z: 0 }, color: '#00ff00', points: new Array(2) } as unknown as ClusterData],
      ['Poetry', { centroid: { x: 10, y: 10, z: 0 }, color: '#ff0000', points: new Array(5) } as unknown as ClusterData],
    ]);
    const { ctx, fonts } = measurer();
    const placed = ZoomBandPlacement.forClusters(clusters, ctx, initial);
    expect(fonts).toEqual([LabelFonts.cluster(13), LabelFonts.cluster(13)]);
    expect(placed.map(p => [p.label, p.color, p.dataX, p.priority, p.fontSize])).toEqual([
      ['Poetry', '#ff0000', 10, 5, 13],
      ['Physics', '#00ff00', 90, 2, 13],
    ]);
    expect(placed.every(p => p.placement !== null)).toBe(true);
  });

  it('places nothing for no clusters', () => {
    expect(ZoomBandPlacement.forClusters(new Map(), measurer().ctx, initial)).toEqual([]);
  });
});

describe('ZoomBandPlacement.visibleAt', () => {
  const always: Placement = { minScale: 0, maxScale: 2 };
  const zoomedIn: Placement = { minScale: 0, maxScale: 0.5 };
  const placed = (label: string, dataX: number, dataY: number, placement: Placement | null, extra: Partial<ClusterLabelPlacement> = {}): ClusterLabelPlacement =>
    ({ label, color: '#ff0000', dataX, dataY, textWidth: 10, placement, priority: 1, ...extra });

  it('draws the labels whose zoom band holds the current zoom', () => {
    const labels = [placed('wide', 40, 40, always), placed('zoomed', 60, 60, zoomedIn), placed('never', 50, 50, null)];
    expect(ZoomBandPlacement.visibleAt(labels, initial, initial.ranges, false).map(d => d.text)).toEqual(['wide']);
    // Half the span on each axis: scale 0.5
    const zoomed = new CartesianView({ xRange: [25, 75], yRange: [25, 75] }, initial.plotArea);
    expect(ZoomBandPlacement.visibleAt(labels, zoomed, initial.ranges, false).map(d => d.text)).toEqual(['wide', 'zoomed']);
  });

  it('hides labels more than 20 px outside the plot area', () => {
    // 800 px over x 0..100: 20 px is 2.5 data units
    const labels = [placed('inside', 50, 50, always), placed('just outside', -2, 50, always), placed('far outside', -3, 50, always)];
    expect(ZoomBandPlacement.visibleAt(labels, initial, initial.ranges, false).map(d => d.text)).toEqual(['inside', 'just outside']);
  });

  it('with "show all", draws every label whatever its zoom band, in the (priority) order given', () => {
    const labels = [
      placed('big', 60, 60, always, { priority: 9 }),
      placed('never', 50, 50, null, { priority: 5 }),
      placed('small', 40, 40, zoomedIn, { priority: 1 }),
    ];
    expect(ZoomBandPlacement.visibleAt(labels, initial, initial.ranges, true).map(d => d.text)).toEqual(['big', 'never', 'small']);
  });

  it('places, sizes and colours each label and names what a click on it reports', () => {
    const [centroid, topic, summary] = ZoomBandPlacement.visibleAt([
      placed('Poetry', 50, 50, always),
      placed('Physics', 25, 75, always, { fontSize: 14, topicLabel: 'Physics' }),
      placed('words-like-these', 75, 25, always, { fontSize: 12, topicLabel: null }),
    ], initial, initial.ranges, false);
    expect(centroid).toEqual({ text: 'Poetry', x: 400, y: 300, width: 10, fontSize: 13, color: '#ff0000', alpha: 1, hitLabel: 'Poetry' });
    expect(topic).toMatchObject({ x: 200, y: 150, fontSize: 14, hitLabel: 'Physics' });
    expect(summary).toMatchObject({ fontSize: 12, hitLabel: null });
  });
});
