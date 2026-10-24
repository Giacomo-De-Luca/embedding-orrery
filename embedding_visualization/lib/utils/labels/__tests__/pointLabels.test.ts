import { describe, expect, it } from 'vitest';

import { CollisionGrid } from '../../collisionGrid';
import { LabelFonts, LabelStyle } from '../labelStyle';
import { PointLabelPass, PointLabelRanking, type PointLabel } from '../pointLabels';
import { RecordingContext } from './recordingContext';

interface TestPoint {
  index: number;
  id: string;
  label?: string;
  x: number;
  y: number;
}

const point = (index: number, x: number, y: number, label?: string): TestPoint => ({ index, id: `id${index}`, label, x, y });
const project = (p: TestPoint) => ({ x: p.x, y: p.y });
const style = new LabelStyle(true, '3d');

/** A labelled point at (x, y) with the given text and similarity. */
const label = (text: string, x: number, y: number, similarity = 1, selected = false): PointLabel<TestPoint> =>
  ({ point: point(0, x, y, text), text, similarity, selected });

describe('PointLabelRanking', () => {
  it('puts the selected point first, then the rest by similarity, keeping ties in order', () => {
    const points = [point(1, 0, 0, 'a'), point(2, 0, 0, 'b'), point(3, 0, 0, 'c'), point(4, 0, 0, 'd')];
    const similarities = new Map([[1, 0.5], [2, 0.9], [3, 0.9], [4, 0.1]]);
    const ranked = PointLabelRanking.rank(points, similarities, 4);
    expect(ranked.selected).toMatchObject({ text: 'd', similarity: 0.1, selected: true });
    expect(ranked.others.map(l => l.text)).toEqual(['b', 'c', 'a']);
    expect(ranked.others.every(l => !l.selected)).toBe(true);
  });

  it('has no selected label when the selected point is not highlighted', () => {
    const ranked = PointLabelRanking.rank([point(1, 0, 0, 'a')], new Map([[1, 0.5]]), 7);
    expect(ranked.selected).toBeNull();
    expect(ranked.others).toHaveLength(1);
  });

  it('labels a point by its label, else its id', () => {
    const ranked = PointLabelRanking.rank([point(1, 0, 0), point(2, 0, 0, 'named')], new Map([[1, 0.2], [2, 0.1]]), null);
    expect(ranked.others.map(l => l.text)).toEqual(['id1', 'named']);
  });

  it('measures every label once in the point font when given a measurer', () => {
    const measurer = new RecordingContext();
    const points = [point(1, 0, 0, 'abc'), point(2, 0, 0, 'de')];
    const ranked = PointLabelRanking.rank(points, new Map([[1, 0.5], [2, 0.9]]), 1, measurer);
    expect(measurer.font).toBe(LabelFonts.point());
    expect(measurer.measured).toEqual(['abc', 'de']);
    expect(ranked.selected?.width).toBe(21);
    expect(ranked.others.map(l => l.width)).toEqual([14]);
  });

  it('leaves the widths to the canvas without a measurer', () => {
    const ranked = PointLabelRanking.rank([point(1, 0, 0, 'abc')], new Map([[1, 0.5]]), null, null);
    expect(ranked.others[0].width).toBeUndefined();
  });
});

describe('PointLabelPass', () => {
  it('draws a label 8 px right of its point exactly as 3D does: outline, then fill', () => {
    const ctx = new RecordingContext();
    const grid = new CollisionGrid({ loX: 0, loY: 0, hiX: 400, hiY: 400 }, 10, 10);
    const drawn = new PointLabelPass(style).draw(ctx, grid, [label('Alpha', 100, 50, 0.5)], project, 8);

    // probe {x+8, y−6.6, x+9, y+6.6}, then the box {x+6, y−7.6, x+10+w, y+7.6} with w = 35
    expect(drawn).toHaveLength(1);
    expect(drawn[0].loX).toBeCloseTo(106);
    expect(drawn[0].loY).toBeCloseTo(42.4);
    expect(drawn[0].hiX).toBeCloseTo(145);
    expect(drawn[0].hiY).toBeCloseTo(57.6);
    expect(ctx.draws.map(d => [d.op, d.text, d.x, d.y])).toEqual([
      ['stroke', 'Alpha', 108, 50],
      ['fill', 'Alpha', 108, 50],
    ]);
    const [stroke, fill] = ctx.draws;
    expect(stroke).toMatchObject({ font: LabelFonts.point(), textAlign: 'left', textBaseline: 'middle' });
    expect(stroke).toMatchObject({ color: style.pointStroke.color, lineWidth: style.pointStroke.width });
    expect(fill.color).toBe(style.pointFill);
    expect(fill.alpha).toBeCloseTo(LabelStyle.pointAlpha(0.5, false));
    expect(ctx.globalAlpha).toBe(1);
  });

  it('keeps the selected label opaque', () => {
    const ctx = new RecordingContext();
    const grid = CollisionGrid.forViewport(400, 400);
    new PointLabelPass(style).draw(ctx, grid, [label('Chosen', 100, 50, 0, true)], project, 8);
    expect(ctx.draws[1].alpha).toBe(1);
  });

  it('skips a label whose start is taken without measuring its text', () => {
    const ctx = new RecordingContext();
    const grid = CollisionGrid.forViewport(400, 400);
    grid.insert({ loX: 107, loY: 45, hiX: 110, hiY: 55 });
    const drawn = new PointLabelPass(style).draw(ctx, grid, [label('Hidden', 100, 50)], project, 8);
    expect(drawn).toEqual([]);
    expect(ctx.measured).toEqual([]);
    expect(ctx.draws).toEqual([]);
  });

  it('skips a label whose text would overlap a box already in the grid', () => {
    const ctx = new RecordingContext();
    const grid = CollisionGrid.forViewport(400, 400);
    grid.insert({ loX: 130, loY: 45, hiX: 140, hiY: 55 });
    const drawn = new PointLabelPass(style).draw(ctx, grid, [label('Blocked', 100, 50)], project, 8);
    expect(drawn).toEqual([]);
    expect(ctx.measured).toEqual(['Blocked']);
    expect(ctx.draws).toEqual([]);
  });

  it('uses a label\'s known width instead of measuring it again', () => {
    const ctx = new RecordingContext();
    const grid = CollisionGrid.forViewport(400, 400);
    const drawn = new PointLabelPass(style).draw(ctx, grid, [{ ...label('Alpha', 100, 50), width: 50 }], project, 8);
    expect(ctx.measured).toEqual([]);
    expect(drawn[0].hiX).toBeCloseTo(160);
    expect(ctx.filled).toEqual(['Alpha']);
  });

  it('skips points that do not project and draws the first of two that overlap', () => {
    const ctx = new RecordingContext();
    const grid = CollisionGrid.forViewport(400, 400);
    const labels = [label('Behind', 0, 0), label('First', 100, 50), label('Second', 100, 52), label('Apart', 100, 200)];
    new PointLabelPass(style).draw(ctx, grid, labels, p => (p.label === 'Behind' ? null : project(p)), 8);
    expect(ctx.filled).toEqual(['First', 'Apart']);
  });

  it('offsets labels beside a marker by its radius plus a 4 px gap', () => {
    expect(PointLabelPass.offsetBeside(10)).toBe(9);
    expect(PointLabelPass.offsetBeside(0)).toBe(4);
  });
});
