import type { BoundingBox } from '../collisionGrid';

/**
 * Screen-space boxes of canvas labels, shared by the collision tests and the
 * zoom-band placement so every label is sized the same way. Labels are drawn
 * with `textBaseline: 'middle'`, so text spans 0.6 × its font size above and
 * below its anchor.
 */
export class LabelBox {
  /** Padding around a cluster label. */
  static readonly PAD = 4;
  /** Half the text height, as a fraction of the font size. */
  static readonly HALF_HEIGHT = 0.6;

  /** A box centred on (cx, cy) around text `width` px wide. */
  static centered(cx: number, cy: number, width: number, fontSize: number, pad = LabelBox.PAD): BoundingBox {
    const halfWidth = width / 2 + pad;
    const halfHeight = fontSize * LabelBox.HALF_HEIGHT + pad;
    return { loX: cx - halfWidth, loY: cy - halfHeight, hiX: cx + halfWidth, hiY: cy + halfHeight };
  }

  /** The box of text starting `offset` px right of (x, y), with 2 px side and 1 px vertical pads. */
  static rightOf(x: number, y: number, offset: number, width: number, fontSize: number): BoundingBox {
    const halfHeight = fontSize * LabelBox.HALF_HEIGHT;
    return { loX: x + offset - 2, loY: y - halfHeight - 1, hiX: x + offset + width + 2, hiY: y + halfHeight + 1 };
  }

  /**
   * A 1 px strip where text starting `offset` px right of (x, y) would begin: tested
   * before the text is measured, so labels in crowded areas are rejected cheaply.
   */
  static probe(x: number, y: number, offset: number, fontSize: number): BoundingBox {
    const halfHeight = fontSize * LabelBox.HALF_HEIGHT;
    return { loX: x + offset, loY: y - halfHeight, hiX: x + offset + 1, hiY: y + halfHeight };
  }

  static isFinite(box: BoundingBox): boolean {
    return Number.isFinite(box.loX) && Number.isFinite(box.loY)
      && Number.isFinite(box.hiX) && Number.isFinite(box.hiY);
  }
}
