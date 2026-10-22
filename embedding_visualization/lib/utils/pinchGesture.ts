/** A finger on the plot, in client pixels (`Touch.identifier`, `clientX`, `clientY`). */
export interface TouchPoint {
  id: number;
  x: number;
  y: number;
}

/** The tracked pair of fingers. */
export interface PinchFrame {
  /** Finger spread ÷ the spread when the pinch began: above 1 the fingers moved apart. */
  scale: number;
  /** Midpoint between the two fingers, in client pixels. */
  centerX: number;
  centerY: number;
}

/** `touchstart`, `touchmove`, or `touchend` / `touchcancel`. */
export type TouchPhase = 'start' | 'move' | 'end';

/** What to do with one touch event. */
export interface PinchStep {
  /** Keep the event from Plotly (`preventDefault` + `stopPropagation`). */
  consume: boolean;
  /** The pair after this event, or null when no pinch is running. */
  frame: PinchFrame | null;
  /** This event started a pinch (its frame has scale 1). */
  started: boolean;
  /** This event lifted a finger of the pair. */
  ended: boolean;
}

/** Spreads below this count as this, so a pinch from coincident fingers never divides by zero. */
const MIN_SPREAD_PX = 1;

const spread = (a: TouchPoint, b: TouchPoint) => Math.max(Math.hypot(a.x - b.x, a.y - b.y), MIN_SPREAD_PX);

/**
 * Two-finger pinch recognition for the scatter plots, free of the DOM.
 *
 * One finger is left to Plotly (2D pan, 3D rotation). From the moment a second
 * finger lands the gesture belongs to the pinch until every finger has lifted:
 * a finger left down after a pinch is swallowed too, because Plotly would resume
 * its pan or rotation from where that finger was before the pinch and jump. Only
 * the last lift reaches Plotly, so it can release its own drag. While the gesture
 * is the pinch's, any two fingers on the plot pinch: a finger put back down starts
 * a fresh pinch, and so does lifting one of the pair while two others remain.
 *
 * `touches` is every finger still on the plot after the event, as in `TouchEvent.touches`.
 */
export class PinchGesture {
  private pair: [number, number] | null = null;
  private startSpread = MIN_SPREAD_PX;
  /** True from a pinch's start until the last finger lifts. */
  private owned = false;

  update(phase: TouchPhase, touches: readonly TouchPoint[]): PinchStep {
    let ended = false;
    if (this.pair && !this.fingers(touches)) {
      this.pair = null;
      ended = true;
    }

    let started = false;
    if (!this.pair && touches.length >= 2 && (phase === 'start' || this.owned)) {
      this.pair = [touches[0].id, touches[1].id];
      this.startSpread = spread(touches[0], touches[1]);
      this.owned = true;
      started = true;
    }

    const fingers = this.fingers(touches);
    const frame = fingers && {
      scale: spread(...fingers) / this.startSpread,
      centerX: (fingers[0].x + fingers[1].x) / 2,
      centerY: (fingers[0].y + fingers[1].y) / 2,
    };

    const released = touches.length === 0;
    const consume = this.owned && !released;
    if (released) this.owned = false;
    return { consume, frame, started, ended };
  }

  /** The tracked pair among `touches`, or null when either has lifted. */
  private fingers(touches: readonly TouchPoint[]): [TouchPoint, TouchPoint] | null {
    const pair = this.pair;
    if (!pair) return null;
    const a = touches.find(t => t.id === pair[0]);
    const b = touches.find(t => t.id === pair[1]);
    return a && b ? [a, b] : null;
  }
}
