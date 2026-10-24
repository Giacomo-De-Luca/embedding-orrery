import type { LabelPaintContext } from '../labelCanvas';

/** One text draw, with the canvas state it was drawn with. */
export interface TextDraw {
  op: 'stroke' | 'fill';
  text: string;
  x: number;
  y: number;
  font: string;
  alpha: number;
  color: unknown;
  lineWidth: number;
  textAlign: string;
  textBaseline: string;
}

/**
 * A canvas context for label tests (node has no canvas): text is 7 px per
 * character, and every measurement and text draw is recorded in order.
 */
export class RecordingContext implements LabelPaintContext {
  font = '';
  globalAlpha = 1;
  fillStyle: string | CanvasGradient | CanvasPattern = '';
  strokeStyle: string | CanvasGradient | CanvasPattern = '';
  lineWidth = 1;
  lineJoin: CanvasLineJoin = 'miter';
  textAlign: CanvasTextAlign = 'start';
  textBaseline: CanvasTextBaseline = 'alphabetic';

  readonly measured: string[] = [];
  readonly draws: TextDraw[] = [];

  measureText(text: string) {
    this.measured.push(text);
    return { width: text.length * 7 };
  }

  strokeText(text: string, x: number, y: number) {
    this.record('stroke', text, x, y, this.strokeStyle);
  }

  fillText(text: string, x: number, y: number) {
    this.record('fill', text, x, y, this.fillStyle);
  }

  /** The texts that were filled, in drawing order. */
  get filled(): string[] {
    return this.draws.filter(d => d.op === 'fill').map(d => d.text);
  }

  private record(op: TextDraw['op'], text: string, x: number, y: number, color: unknown) {
    this.draws.push({
      op, text, x, y, color,
      font: this.font,
      alpha: this.globalAlpha,
      lineWidth: this.lineWidth,
      textAlign: this.textAlign,
      textBaseline: this.textBaseline,
    });
  }
}
