import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CameraPinchZoom, CartesianPinchZoom, PinchZoomSession, type FrameScheduler, type PinchZoomTarget } from '../plotPinchZoom';

/** Plotly's 2D wheel zoom: the axis span is multiplied by exp(clamp(deltaY, ±20) / 200) per event. */
const plotlySpanFactor = (deltas: number[]) =>
  deltas.reduce((factor, d) => factor * Math.exp(Math.min(Math.max(d, -20), 20) / 200), 1);

const frame = (scale: number, centerX = 150, centerY = 100) => ({ scale, centerX, centerY });

describe('CartesianPinchZoom.wheelDeltas', () => {
  it('returns nothing when the span does not change', () => {
    expect(CartesianPinchZoom.wheelDeltas(1)).toEqual([]);
  });

  it('zooms in with negative deltas and out with positive ones', () => {
    expect(CartesianPinchZoom.wheelDeltas(0.95)[0]).toBeLessThan(0);
    expect(CartesianPinchZoom.wheelDeltas(1.05)[0]).toBeGreaterThan(0);
  });

  it('reproduces the requested span ratio through Plotly\'s wheel formula', () => {
    for (const ratio of [0.97, 1.04, 0.5, 3, 0.1]) {
      expect(plotlySpanFactor(CartesianPinchZoom.wheelDeltas(ratio))).toBeCloseTo(ratio, 10);
    }
  });

  it('splits large ratios into events Plotly does not clamp', () => {
    const deltas = CartesianPinchZoom.wheelDeltas(0.25);
    expect(deltas.length).toBeGreaterThan(1);
    for (const d of deltas) expect(Math.abs(d)).toBeLessThanOrEqual(CartesianPinchZoom.MAX_WHEEL_DELTA);
  });
});

describe('CartesianPinchZoom', () => {
  interface RecordedWheel { type: string; deltaY: number; clientX: number; clientY: number; bubbles?: boolean; cancelable?: boolean }

  let wheels: RecordedWheel[];
  let dragger: { dispatchEvent: (e: Event) => boolean };

  beforeEach(() => {
    wheels = [];
    dragger = { dispatchEvent: (e) => { wheels.push(e as unknown as RecordedWheel); return true; } };
    // The node test environment has no WheelEvent; record its type and init dict instead.
    vi.stubGlobal('WheelEvent', class {
      constructor(type: string, init: Omit<RecordedWheel, 'type'>) {
        return { type, ...init };
      }
    });
  });

  afterEach(() => vi.unstubAllGlobals());

  const graphDiv = (drag: Partial<{ _dragging: boolean; _dragged: boolean; _dragdata: unknown }> = {}) => ({
    querySelector: (selector: string) => (selector === '.nsewdrag' ? dragger : null),
    ...drag,
  });

  it('replays the spread change since the last frame as wheel events at the midpoint', () => {
    const zoom = new CartesianPinchZoom({ current: graphDiv() });
    zoom.start(frame(1));
    zoom.pinch(frame(1.25, 120, 80));
    zoom.pinch(frame(1.5, 130, 90));

    expect(wheels.every(w => w.type === 'wheel' && w.cancelable)).toBe(true);
    const first = wheels.filter(w => w.clientX === 120 && w.clientY === 80);
    const second = wheels.filter(w => w.clientX === 130 && w.clientY === 90);
    expect(first.length + second.length).toBe(wheels.length);
    // Fingers apart zoom in: the span shrinks by 1 / scale overall, frame by frame.
    expect(plotlySpanFactor(first.map(w => w.deltaY))).toBeCloseTo(1 / 1.25, 10);
    expect(plotlySpanFactor(second.map(w => w.deltaY))).toBeCloseTo(1.25 / 1.5, 10);
  });

  it('hands Plotly\'s one-finger drag over to the pinch', () => {
    const gd = graphDiv({ _dragging: true, _dragged: false, _dragdata: { dx: 3, dy: 0 } });
    const zoom = new CartesianPinchZoom({ current: gd });
    zoom.start(frame(1));
    expect(gd._dragging).toBe(false);
    expect('_dragdata' in gd).toBe(false);

    zoom.end();
    expect(wheels).toEqual([]);
  });

  it('commits a pan Plotly made before the pinch with a zero wheel at the end', () => {
    const zoom = new CartesianPinchZoom({ current: graphDiv({ _dragging: true, _dragged: true }) });
    zoom.start(frame(1, 140, 60));
    zoom.end();
    expect(wheels).toEqual([{ type: 'wheel', deltaY: 0, clientX: 140, clientY: 60, bubbles: false, cancelable: true }]);
  });

  it('does nothing before Plotly has drawn its drag surface', () => {
    const zoom = new CartesianPinchZoom({ current: null });
    zoom.start(frame(1));
    zoom.pinch(frame(2));
    zoom.end();
    expect(wheels).toEqual([]);
  });
});

/**
 * gl-plot3d camera stand-in. Writes are clamped to the fork's `zoomMax` (2) and
 * become the view's latest keyframe; the `distance` getter only catches up when
 * the scene renders, as in the real camera.
 */
class FakeCamera {
  private latest: number;
  private rendered: number;
  readonly view = {
    computedRadius: [0],
    lastT: () => 0,
    recalcMatrix: () => { this.view.computedRadius[0] = Math.log(this.latest); },
  };

  constructor(distance: number) {
    this.latest = this.rendered = distance;
  }

  get distance() { return this.rendered; }
  set distance(d: number) { this.latest = Math.min(d, 2); }

  /** The distance the next render will show. */
  get target() { return this.latest; }
  render() { this.rendered = this.latest; }
}

describe('CameraPinchZoom', () => {
  const sceneGraphDiv = (camera: FakeCamera | undefined) => ({
    _fullLayout: { scene: { _scene: { glplot: { camera } } } },
  });

  it('scales the camera distance from where the pinch started', () => {
    const camera = new FakeCamera(1.5);
    const zoom: PinchZoomTarget = new CameraPinchZoom({ current: sceneGraphDiv(camera) });
    zoom.start(frame(1));

    zoom.pinch(frame(2));
    expect(camera.target).toBeCloseTo(0.75);
    zoom.pinch(frame(0.8));
    expect(camera.target).toBeCloseTo(1.875);
  });

  it('leaves the zoom-out limit to the camera', () => {
    const camera = new FakeCamera(1.5);
    const zoom: PinchZoomTarget = new CameraPinchZoom({ current: sceneGraphDiv(camera) });
    zoom.start(frame(1));
    zoom.pinch(frame(0.5));
    expect(camera.target).toBe(2);
  });

  it('starts each pinch from the camera\'s latest distance, even before it has rendered', () => {
    const camera = new FakeCamera(1.5);
    const zoom: PinchZoomTarget = new CameraPinchZoom({ current: sceneGraphDiv(camera) });
    zoom.start(frame(1));
    zoom.pinch(frame(1.5));
    zoom.end();

    expect(camera.distance).toBe(1.5); // not rendered yet
    zoom.start(frame(1));
    zoom.pinch(frame(2));
    expect(camera.target).toBeCloseTo(0.5);
  });

  it('ignores a pinch when the 3D scene is not ready', () => {
    const zoom: PinchZoomTarget = new CameraPinchZoom({ current: sceneGraphDiv(undefined) });
    expect(() => {
      zoom.start(frame(1));
      zoom.pinch(frame(2));
      zoom.end();
    }).not.toThrow();
  });
});

/** Animation frames that run only when the test says so. */
class FakeFrames implements FrameScheduler {
  private readonly callbacks = new Map<number, () => void>();
  private nextId = 1;

  request(callback: () => void) {
    const id = this.nextId++;
    this.callbacks.set(id, callback);
    return id;
  }

  cancel(id: number) { this.callbacks.delete(id); }

  get waiting() { return this.callbacks.size; }

  run() {
    const callbacks = [...this.callbacks.values()];
    this.callbacks.clear();
    for (const callback of callbacks) callback();
  }
}

describe('PinchZoomSession', () => {
  const finger = (id: number, x: number, y: number) => ({ id, x, y });
  const pair = (spread: number) => [finger(1, 100, 100), finger(2, 100 + spread, 100)];

  let calls: string[];
  let frames: FakeFrames;
  let session: PinchZoomSession;

  beforeEach(() => {
    calls = [];
    frames = new FakeFrames();
    const target: PinchZoomTarget = {
      start: f => calls.push(`start ${f.scale}`),
      pinch: f => calls.push(`pinch ${f.scale}`),
      end: () => calls.push('end'),
    };
    session = new PinchZoomSession(target, frames);
  });

  it('leaves one finger alone', () => {
    expect(session.handle('start', [finger(1, 0, 0)])).toBe(false);
    expect(session.handle('move', [finger(1, 20, 0)])).toBe(false);
    expect(session.handle('end', [])).toBe(false);
    expect(calls).toEqual([]);
    expect(frames.waiting).toBe(0);
  });

  it('starts the target when the second finger lands and keeps the event from Plotly', () => {
    session.handle('start', [finger(1, 100, 100)]);
    expect(session.handle('start', pair(100))).toBe(true);
    expect(calls).toEqual(['start 1']);
  });

  it('applies the latest of several moves once per animation frame', () => {
    session.handle('start', pair(100));
    expect(session.handle('move', pair(150))).toBe(true);
    session.handle('move', pair(200));
    expect(frames.waiting).toBe(1);
    expect(calls).toEqual(['start 1']);

    frames.run();
    expect(calls).toEqual(['start 1', 'pinch 2']);
  });

  it('applies a waiting move before ending, and nothing after', () => {
    session.handle('start', pair(100));
    session.handle('move', pair(300));
    expect(session.handle('end', [finger(1, 100, 100)])).toBe(true);
    expect(calls).toEqual(['start 1', 'pinch 3', 'end']);

    session.handle('move', [finger(1, 150, 100)]);
    frames.run();
    expect(calls).toEqual(['start 1', 'pinch 3', 'end']);
    expect(session.handle('end', [])).toBe(false);
  });

  it('drops a waiting move when disposed', () => {
    session.handle('start', pair(100));
    session.handle('move', pair(200));
    session.dispose();
    expect(frames.waiting).toBe(0);
    expect(calls).toEqual(['start 1']);
  });
});
