import { describe, expect, it } from 'vitest';
import { PinchGesture, type TouchPoint } from '../pinchGesture';

const finger = (id: number, x: number, y: number): TouchPoint => ({ id, x, y });

/** A pinch whose two fingers start 100 px apart around (150, 100). */
function startedPinch() {
  const gesture = new PinchGesture();
  gesture.update('start', [finger(1, 100, 100)]);
  const step = gesture.update('start', [finger(1, 100, 100), finger(2, 200, 100)]);
  return { gesture, step };
}

describe('PinchGesture', () => {
  it('leaves one finger to Plotly', () => {
    const gesture = new PinchGesture();
    for (const [phase, touches] of [
      ['start', [finger(1, 10, 10)]],
      ['move', [finger(1, 40, 10)]],
      ['end', []],
    ] as const) {
      expect(gesture.update(phase, touches)).toEqual({ consume: false, frame: null, started: false, ended: false });
    }
  });

  it('starts a pinch at scale 1 when the second finger lands', () => {
    const { step } = startedPinch();
    expect(step).toEqual({
      consume: true,
      frame: { scale: 1, centerX: 150, centerY: 100 },
      started: true,
      ended: false,
    });
  });

  it('reports the spread relative to the start and the midpoint', () => {
    const { gesture } = startedPinch();
    const apart = gesture.update('move', [finger(1, 50, 100), finger(2, 250, 100)]);
    expect(apart).toMatchObject({ consume: true, started: false, frame: { scale: 2, centerX: 150, centerY: 100 } });

    const together = gesture.update('move', [finger(1, 125, 140), finger(2, 175, 140)]);
    expect(together.frame).toEqual({ scale: 0.5, centerX: 150, centerY: 140 });
  });

  it('keeps tracking the original pair when a third finger lands', () => {
    const { gesture } = startedPinch();
    const third = gesture.update('start', [finger(1, 100, 100), finger(2, 200, 100), finger(3, 0, 0)]);
    expect(third).toMatchObject({ consume: true, started: false, ended: false });

    const move = gesture.update('move', [finger(3, 0, 0), finger(1, 50, 100), finger(2, 250, 100)]);
    expect(move.frame).toEqual({ scale: 2, centerX: 150, centerY: 100 });
  });

  it('ends the pinch when a finger of the pair lifts and swallows the finger left behind', () => {
    const { gesture } = startedPinch();
    expect(gesture.update('end', [finger(1, 100, 100)])).toEqual({ consume: true, frame: null, started: false, ended: true });
    expect(gesture.update('move', [finger(1, 160, 130)])).toEqual({ consume: true, frame: null, started: false, ended: false });
  });

  it('lets the last lift through and returns to one-finger mode', () => {
    const { gesture } = startedPinch();
    gesture.update('end', [finger(1, 100, 100)]);
    expect(gesture.update('end', [])).toEqual({ consume: false, frame: null, started: false, ended: false });

    expect(gesture.update('start', [finger(4, 10, 10)]).consume).toBe(false);
    expect(gesture.update('move', [finger(4, 30, 10)]).consume).toBe(false);
  });

  it('ends and releases at once when both fingers lift together', () => {
    const { gesture } = startedPinch();
    expect(gesture.update('end', [])).toEqual({ consume: false, frame: null, started: false, ended: true });
  });

  it('starts a fresh pinch when a finger lands again while one remains', () => {
    const { gesture } = startedPinch();
    gesture.update('end', [finger(2, 200, 100)]);
    const again = gesture.update('start', [finger(2, 200, 100), finger(5, 200, 300)]);
    expect(again).toEqual({ consume: true, frame: { scale: 1, centerX: 200, centerY: 200 }, started: true, ended: false });

    const move = gesture.update('move', [finger(2, 200, 50), finger(5, 200, 350)]);
    expect(move.frame?.scale).toBeCloseTo(1.5);
  });

  it('pinches on with the two fingers left when one of the pair lifts beside a third', () => {
    const { gesture } = startedPinch();
    gesture.update('start', [finger(1, 100, 100), finger(2, 200, 100), finger(3, 100, 300)]);
    const lift = gesture.update('end', [finger(2, 200, 100), finger(3, 100, 300)]);
    expect(lift).toEqual({ consume: true, frame: { scale: 1, centerX: 150, centerY: 200 }, started: true, ended: true });
  });

  it('only starts a pinch on a finger landing, never mid-move', () => {
    const gesture = new PinchGesture();
    const step = gesture.update('move', [finger(1, 0, 0), finger(2, 100, 0)]);
    expect(step).toEqual({ consume: false, frame: null, started: false, ended: false });
  });

  it('never divides by zero when the fingers start on the same spot', () => {
    const gesture = new PinchGesture();
    gesture.update('start', [finger(1, 100, 100)]);
    gesture.update('start', [finger(1, 100, 100), finger(2, 100, 100)]);
    const step = gesture.update('move', [finger(1, 95, 100), finger(2, 105, 100)]);
    expect(Number.isFinite(step.frame?.scale)).toBe(true);
    expect(step.frame?.scale).toBeGreaterThan(1);
  });
});
