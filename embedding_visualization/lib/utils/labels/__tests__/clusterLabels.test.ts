import { describe, expect, it } from 'vitest';

import { CollisionGrid } from '../../collisionGrid';
import { ClusterLabelOrder, ClusterLabelPainter, type ClusterLabelDraw } from '../clusterLabels';
import { LabelHitBoxes } from '../labelHitBoxes';
import { LabelFonts, LabelStyle } from '../labelStyle';
import { RecordingContext } from './recordingContext';

interface TestCluster {
  name: string;
  pointCount: number;
  distance: number;
}

const clusters = (distances: number[]): TestCluster[] =>
  distances.map((distance, i) => ({ name: `c${i}`, pointCount: 10, distance }));

const draw = (text: string, x: number, y: number, extra: Partial<ClusterLabelDraw> = {}): ClusterLabelDraw => ({
  text, x, y, fontSize: 13, color: '#ff0000', alpha: 1, hitLabel: text, ...extra,
});

describe('ClusterLabelOrder.nearest (3D)', () => {
  it('takes the nearest max(8, ⌊2.5·√n⌋) clusters, fading from 1 to 0.65 with distance', () => {
    const labels = clusters(Array.from({ length: 20 }, (_, i) => 19 - i));
    const chosen = ClusterLabelOrder.nearest(labels, c => c.distance, false);
    expect(chosen).toHaveLength(11);
    expect(chosen.map(c => c.label.distance)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(chosen[0].alpha).toBe(1);
    expect(chosen[10].alpha).toBeCloseTo(0.65);
    expect(chosen[5].alpha).toBeCloseTo(1 - 0.35 * 0.5);
  });

  it('keeps at least 8 and breaks distance ties by point count', () => {
    const labels: TestCluster[] = [
      { name: 'small', pointCount: 2, distance: 1 },
      { name: 'big', pointCount: 9, distance: 1 },
      { name: 'far', pointCount: 50, distance: 3 },
    ];
    expect(ClusterLabelOrder.nearest(labels, c => c.distance, false).map(c => c.label.name)).toEqual(['big', 'small', 'far']);
  });

  it('with "show all", takes every cluster in its order at full opacity', () => {
    const labels = clusters(Array.from({ length: 20 }, (_, i) => i));
    const chosen = ClusterLabelOrder.nearest(labels, c => c.distance, true);
    expect(chosen).toHaveLength(20);
    expect(chosen.map(c => c.label.name)).toEqual(labels.map(c => c.name));
    expect(chosen.every(c => c.alpha === 1)).toBe(true);
  });
});

describe('ClusterLabelOrder.byPriority (2D)', () => {
  it('puts the highest priority first, keeping ties in order', () => {
    const labels = [{ name: 'a', priority: 1 }, { name: 'b', priority: 5 }, { name: 'c', priority: 5 }];
    expect(ClusterLabelOrder.byPriority(labels).map(l => l.name)).toEqual(['b', 'c', 'a']);
    expect(labels.map(l => l.name)).toEqual(['a', 'b', 'c']);
  });
});

describe('ClusterLabelPainter', () => {
  const style = new LabelStyle(false, '2d');

  it('draws a label centred on its position, outline first, and records its hit box', () => {
    const ctx = new RecordingContext();
    const hits = new LabelHitBoxes();
    new ClusterLabelPainter(style).paint(ctx, CollisionGrid.forViewport(400, 400), [draw('Poetry', 100, 50, { alpha: 0.8 })], hits);

    expect(ctx.draws.map(d => [d.op, d.text, d.x, d.y])).toEqual([['stroke', 'Poetry', 100, 50], ['fill', 'Poetry', 100, 50]]);
    const [stroke, fill] = ctx.draws;
    expect(stroke).toMatchObject({ font: LabelFonts.cluster(13), textAlign: 'center', textBaseline: 'middle' });
    expect(stroke).toMatchObject({ color: style.clusterStroke.color, lineWidth: style.clusterStroke.width });
    expect(fill).toMatchObject({ color: style.clusterFill('#ff0000'), alpha: 0.8 });
    expect(ctx.globalAlpha).toBe(1);
    // Box: half of 42 px + 4 wide, 0.6 × 13 + 4 high
    expect(hits.at(100 - 25, 50)).toEqual(['Poetry']);
    expect(hits.at(100 + 26, 50)).toEqual([]);
  });

  it('uses a label\'s known width instead of measuring it again', () => {
    const ctx = new RecordingContext();
    const hits = new LabelHitBoxes();
    new ClusterLabelPainter(style).paint(ctx, CollisionGrid.forViewport(400, 400), [draw('Poetry', 100, 50, { width: 100 })], hits);
    expect(ctx.measured).toEqual([]);
    // Box: half of 100 px + 4 wide
    expect(hits.at(100 - 54, 50)).toEqual(['Poetry']);
  });

  it('draws each label at its own font size', () => {
    const ctx = new RecordingContext();
    new ClusterLabelPainter(style).paint(ctx, CollisionGrid.forViewport(400, 400), [draw('Coarse', 100, 50, { fontSize: 14 })], new LabelHitBoxes());
    expect(ctx.draws[0].font).toBe(LabelFonts.cluster(14));
  });

  it('skips a label that collides with one drawn before it', () => {
    const ctx = new RecordingContext();
    const hits = new LabelHitBoxes();
    const labels = [draw('First', 100, 50), draw('Second', 110, 55), draw('Apart', 100, 200)];
    new ClusterLabelPainter(style).paint(ctx, CollisionGrid.forViewport(400, 400), labels, hits);
    expect(ctx.filled).toEqual(['First', 'Apart']);
    expect(hits.at(110, 55)).toEqual(['First']);
  });

  it('draws labels that are not clickable without a hit box', () => {
    const ctx = new RecordingContext();
    const hits = new LabelHitBoxes();
    new ClusterLabelPainter(style).paint(ctx, CollisionGrid.forViewport(400, 400), [draw('summary', 100, 50, { hitLabel: null })], hits);
    expect(ctx.filled).toEqual(['summary']);
    expect(hits.at(100, 50)).toEqual([]);
  });
});
