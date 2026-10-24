import { LabelHitBoxes } from './labelHitBoxes';
import type { TextMeasurer } from './labelStyle';

/** Where a label redraw paints: the plot's overlay, or a screenshot's own canvas. */
export interface LabelTarget {
  canvas: HTMLCanvasElement;
  /** Backing-store pixels per CSS pixel. */
  pixelRatio: number;
  /** Collects the boxes of the clickable labels drawn; only the overlay's are hit-tested. */
  hits: LabelHitBoxes;
}

/** The canvas calls the label passes make (a 2D context has them). */
export interface LabelPaintContext extends TextMeasurer {
  globalAlpha: number;
  fillStyle: string | CanvasGradient | CanvasPattern;
  strokeStyle: string | CanvasGradient | CanvasPattern;
  lineWidth: number;
  lineJoin: CanvasLineJoin;
  textAlign: CanvasTextAlign;
  textBaseline: CanvasTextBaseline;
  strokeText(text: string, x: number, y: number): void;
  fillText(text: string, x: number, y: number): void;
}

/** The label overlay canvas of a plot. */
export class LabelCanvas {
  /**
   * Readies the canvas for a redraw: sizes its backing store to the CSS size ×
   * the device pixel ratio (only on a change, since resizing reallocates it),
   * scales the context to CSS pixels and clears it. Null without a 2D context.
   */
  static prepare(canvas: HTMLCanvasElement, cssWidth: number, cssHeight: number, dpr: number): CanvasRenderingContext2D | null {
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    const width = Math.round(cssWidth * dpr);
    const height = Math.round(cssHeight * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    return ctx;
  }

  /** Clears everything drawn, e.g. while a camera animation hides the labels. */
  static clear(canvas: HTMLCanvasElement | null | undefined): void {
    const ctx = canvas?.getContext('2d');
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  /** The plot's overlay as a label target, at the screen's pixel ratio; null before the canvas mounts. */
  static overlay(
    canvas: HTMLCanvasElement | null,
    hits: LabelHitBoxes,
    devicePixelRatio: number = window.devicePixelRatio || 1,
  ): LabelTarget | null {
    return canvas ? { canvas, pixelRatio: devicePixelRatio, hits } : null;
  }

  /**
   * Runs a label redraw on a canvas of its own at `pixelRatio` and returns that
   * canvas, for a screenshot: the overlay keeps its pixels and click targets.
   */
  static renderOffscreen(
    pixelRatio: number,
    render: (target: LabelTarget) => void,
    canvas: HTMLCanvasElement = document.createElement('canvas'),
  ): HTMLCanvasElement {
    render({ canvas, pixelRatio, hits: new LabelHitBoxes() });
    return canvas;
  }
}
