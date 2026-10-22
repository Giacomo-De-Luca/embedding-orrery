/**
 * Coalesces redraw requests into one redraw, run in a microtask: after the
 * synchronous work that asked for it, and before the browser paints that work.
 *
 * The 2D overlays (cluster labels, density map) redraw on every
 * `plotly_relayouting`. A fast pinch frame replays as several wheel events, each
 * redrawing the points and emitting one, all inside a single animation-frame
 * callback (`PinchZoomSession`): they get one overlay redraw, in that same frame.
 * A `requestAnimationFrame` asked for from inside that callback would only run
 * on the next frame, one frame behind the points.
 *
 * The redraw must read live state (Plotly's current ranges): a request made while
 * one is queued is then covered by it, and a request made during the redraw
 * queues another.
 */
export class CoalescedRedraw {
  private queued = false;

  constructor(
    private readonly redraw: () => void,
    private readonly enqueue: (callback: () => void) => void = callback => queueMicrotask(callback),
  ) {}

  /** Asks for a redraw; every request before it runs shares it. */
  request(): void {
    if (this.queued) return;
    this.queued = true;
    this.enqueue(() => this.run());
  }

  /** Drops a queued redraw (its owner is going away). */
  cancel(): void {
    this.queued = false;
  }

  private run() {
    if (!this.queued) return;
    this.queued = false;
    this.redraw();
  }
}
