import { describe, expect, it } from 'vitest';

import { LabelCanvas, type LabelTarget } from '../labelCanvas';
import { LabelHitBoxes } from '../labelHitBoxes';

/** A canvas recording its backing-store resizes and its context's transform and clear calls. */
function fakeCanvas(width = 0, height = 0, hasContext = true) {
  const calls: string[] = [];
  let w = width;
  let h = height;
  const ctx = {
    setTransform: (...args: number[]) => calls.push(`setTransform(${args.join(',')})`),
    clearRect: (...args: number[]) => calls.push(`clearRect(${args.join(',')})`),
  };
  const canvas = {
    get width() { return w; },
    set width(value: number) { calls.push(`width=${value}`); w = value; },
    get height() { return h; },
    set height(value: number) { calls.push(`height=${value}`); h = value; },
    getContext: () => (hasContext ? ctx : null),
  };
  return { canvas: canvas as unknown as HTMLCanvasElement, ctx, calls };
}

describe('LabelCanvas', () => {
  it('sizes the backing store for the pixel ratio, scales to CSS pixels and clears', () => {
    const { canvas, ctx, calls } = fakeCanvas();
    expect(LabelCanvas.prepare(canvas, 400, 300, 2)).toBe(ctx);
    expect(calls).toEqual(['width=800', 'height=600', 'setTransform(2,0,0,2,0,0)', 'clearRect(0,0,400,300)']);
  });

  it('leaves the backing store alone when its size is unchanged', () => {
    const { canvas, calls } = fakeCanvas(601, 450);
    LabelCanvas.prepare(canvas, 400.4, 300, 1.5);
    expect(calls).toEqual(['setTransform(1.5,0,0,1.5,0,0)', 'clearRect(0,0,400.4,300)']);
  });

  it('returns null without a 2D context', () => {
    expect(LabelCanvas.prepare(fakeCanvas(0, 0, false).canvas, 400, 300, 1)).toBeNull();
  });

  it('clears the whole backing store, and ignores a missing canvas', () => {
    const { canvas, calls } = fakeCanvas(800, 600);
    LabelCanvas.clear(canvas);
    expect(calls).toEqual(['clearRect(0,0,800,600)']);
    expect(() => LabelCanvas.clear(null)).not.toThrow();
  });

  it('runs a redraw on a canvas of its own at the export pixel ratio, with empty click targets', () => {
    const { canvas, calls } = fakeCanvas();
    let target: LabelTarget | null = null;
    const result = LabelCanvas.renderOffscreen(4.8, (t) => {
      target = t;
      LabelCanvas.prepare(t.canvas, 800, 500, t.pixelRatio);
    }, canvas);
    expect(result).toBe(canvas);
    expect(target!.canvas).toBe(canvas);
    expect(target!.hits.size).toBe(0);
    expect(calls).toEqual(['width=3840', 'height=2400', 'setTransform(4.8,0,0,4.8,0,0)', 'clearRect(0,0,800,500)']);
  });

  it("targets the overlay at the screen's pixel ratio with the plot's click targets, once it is mounted", () => {
    const { canvas } = fakeCanvas();
    const hits = new LabelHitBoxes();
    expect(LabelCanvas.overlay(canvas, hits, 2)).toEqual({ canvas, pixelRatio: 2, hits });
    expect(LabelCanvas.overlay(canvas, hits, 2)!.hits).toBe(hits);
    expect(LabelCanvas.overlay(null, hits, 2)).toBeNull();
  });
});
