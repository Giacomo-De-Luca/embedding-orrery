import { PinchGesture, type PinchFrame, type TouchPhase, type TouchPoint } from './pinchGesture';

/** A plot that zooms with a two-finger pinch; driven by `usePinchZoom`. */
export interface PinchZoomTarget {
  /** A pinch began (`frame.scale` is 1). */
  start(frame: PinchFrame): void;
  /** The pair moved; called at most once per animation frame. */
  pinch(frame: PinchFrame): void;
  /** A finger of the pair lifted. */
  end(): void;
}

/** A ref to Plotly's graph div, read when a pinch runs (never during render). */
interface GraphDivRef<T> {
  readonly current: T | null | undefined;
}

/** The parts of a 2D graph div a pinch uses: the main drag surface and Plotly's one-finger drag state (`dragelement`). */
interface CartesianGraphDiv {
  querySelector(selectors: string): { dispatchEvent(event: Event): boolean } | null;
  _dragging?: boolean;
  _dragged?: boolean;
  _dragdata?: unknown;
}

/** The parts of gl-plot3d's camera (`glplot.camera`) a pinch uses. */
interface PinchCamera {
  /** Read: the distance at the last render tick. Write: clamped to the camera's zoom limits (the fork's `zoomMax`). */
  distance: number;
  /** The camera's view controller; `computedRadius[0]` is the log of the eye distance. */
  view: { lastT(): number; recalcMatrix(t: number): void; computedRadius: ArrayLike<number> };
}

interface SceneGraphDiv {
  _fullLayout?: { scene?: { _scene?: { glplot?: { camera?: PinchCamera } } } };
}

/**
 * 2D pinch: replays the pinch as wheel events on Plotly's main drag surface, so it
 * takes the wheel-zoom path: a cheap scattergl viewport update per frame, one GUI
 * relayout once it pauses (kept by `uirevision`), `plotly_relayouting` for the
 * density and label overlays, and the `useZoomLimit` zoom-out cap, which catches
 * the events on their way down. The zoom is anchored at the fingers' midpoint.
 */
export class CartesianPinchZoom implements PinchZoomTarget {
  /** Largest |deltaY| Plotly's `zoomWheel` takes per event (`forked/plotly.js/src/plots/cartesian/dragbox.js`). */
  static readonly MAX_WHEEL_DELTA = 20;
  /** `zoomWheel` multiplies the axis span by exp(deltaY / WHEEL_ZOOM_SCALE). */
  static readonly WHEEL_ZOOM_SCALE = 200;

  /** Wheel `deltaY`s that multiply the axis span by `spanRatio` (below 1 zooms in). */
  static wheelDeltas(spanRatio: number): number[] {
    const total = CartesianPinchZoom.WHEEL_ZOOM_SCALE * Math.log(spanRatio);
    if (!Number.isFinite(total) || Math.abs(total) < 1e-9) return [];
    const count = Math.ceil(Math.abs(total) / CartesianPinchZoom.MAX_WHEEL_DELTA);
    return Array.from({ length: count }, () => total / count);
  }

  private dragger: ReturnType<CartesianGraphDiv['querySelector']> = null;
  private lastScale = 1;
  private lastCenter = { x: 0, y: 0 };
  private commitPan = false;

  constructor(private readonly graphDiv: GraphDivRef<CartesianGraphDiv>) {}

  start(frame: PinchFrame): void {
    const gd = this.graphDiv.current;
    this.dragger = gd?.querySelector('.nsewdrag') ?? null;
    this.lastScale = frame.scale;
    this.lastCenter = { x: frame.centerX, y: frame.centerY };
    this.commitPan = false;
    if (!gd?._dragging) return;
    // Hand the first finger's drag over to the pinch: without `_dragging` Plotly's
    // touchend only removes its listeners (no click, no double-click reset), and
    // without `_dragdata` a relayout cannot replay the old pan (`Plots.redrag`).
    // A pan Plotly already drew is committed by the zero wheel in `end`.
    this.commitPan = !!gd._dragged;
    gd._dragging = false;
    delete gd._dragdata;
  }

  pinch(frame: PinchFrame): void {
    for (const deltaY of CartesianPinchZoom.wheelDeltas(this.lastScale / frame.scale)) {
      this.wheel(frame.centerX, frame.centerY, deltaY);
    }
    this.lastScale = frame.scale;
    this.lastCenter = { x: frame.centerX, y: frame.centerY };
  }

  end(): void {
    if (this.commitPan) this.wheel(this.lastCenter.x, this.lastCenter.y, 0);
    this.commitPan = false;
    this.dragger = null;
  }

  private wheel(clientX: number, clientY: number, deltaY: number) {
    // Not bubbling: the capture-phase zoom limit on the plot container still sees it.
    this.dragger?.dispatchEvent(new WheelEvent('wheel', { deltaY, clientX, clientY, bubbles: false, cancelable: true }));
  }
}

/**
 * 3D pinch: scales the camera's eye distance by the finger spread, like the wheel's
 * dolly, and stops at the same limits. The camera change needs no relayout: the
 * scene redraws on its own render loop, and `ScatterPlot3D` reads the live camera
 * back before every `Plotly.react`.
 */
export class CameraPinchZoom implements PinchZoomTarget {
  private startDistance: number | null = null;

  constructor(private readonly graphDiv: GraphDivRef<SceneGraphDiv>) {}

  start(): void {
    const camera = this.camera();
    if (!camera) {
      this.startDistance = null;
      return;
    }
    // `distance` lags the last write until the scene renders; like Plotly's
    // `scene.getCamera`, bring the view up to its latest keyframe first, so a
    // pinch restarting right after another starts where that one left the camera.
    camera.view.recalcMatrix(camera.view.lastT());
    this.startDistance = Math.exp(camera.view.computedRadius[0]);
  }

  pinch(frame: PinchFrame): void {
    const camera = this.camera();
    if (camera && this.startDistance !== null) camera.distance = this.startDistance / frame.scale;
  }

  end(): void {
    this.startDistance = null;
  }

  private camera(): PinchCamera | undefined {
    return this.graphDiv.current?._fullLayout?.scene?._scene?.glplot?.camera;
  }
}

/** Animation-frame scheduling, injected so `PinchZoomSession` runs without a browser. */
export interface FrameScheduler {
  request(callback: () => void): number;
  cancel(id: number): void;
}

/**
 * One plot's pinch handling behind `usePinchZoom`: feeds touch events to a
 * `PinchGesture` and drives `target`, coalescing moves to one `pinch` per animation
 * frame. A frame still waiting is applied before `end`, and nothing reaches
 * `pinch` after it.
 */
export class PinchZoomSession {
  private readonly gesture = new PinchGesture();
  private pending: PinchFrame | null = null;
  private frameId: number | null = null;

  constructor(private readonly target: PinchZoomTarget, private readonly frames: FrameScheduler) {}

  /** Feeds one touch event (`touches`: the fingers still on the plot); true when it must be kept from Plotly. */
  handle(phase: TouchPhase, touches: readonly TouchPoint[]): boolean {
    const step = this.gesture.update(phase, touches);
    if (step.ended) {
      this.flush();
      this.target.end();
    }
    if (step.started && step.frame) {
      this.target.start(step.frame);
    } else if (step.frame) {
      this.pending = step.frame;
      if (this.frameId === null) this.frameId = this.frames.request(() => this.flush());
    }
    return step.consume;
  }

  /** Drops a frame still waiting (the plot is going away). */
  dispose(): void {
    if (this.frameId !== null) this.frames.cancel(this.frameId);
    this.frameId = null;
    this.pending = null;
  }

  private flush() {
    const frame = this.pending;
    this.dispose();
    if (frame) this.target.pinch(frame);
  }
}
