import type { BoundingBox, CollisionGrid } from '../collisionGrid';
import type { LabelPaintContext } from './labelCanvas';
import { LabelBox } from './labelBox';
import { LabelFonts, LabelStyle, type TextMeasurer } from './labelStyle';

/** A point a search label can name. */
export interface LabelPoint {
  index: number;
  id: string;
  label?: string;
}

/** A search-result label: its point, text and similarity, and whether it is the selected point's. */
export interface PointLabel<P> {
  point: P;
  text: string;
  /** The text's width in the point font, when already measured; measured on the canvas otherwise. */
  width?: number;
  similarity: number;
  selected: boolean;
}

/** Search labels in the order they claim space: the selected point's first, then the rest. */
export interface RankedPointLabels<P> {
  selected: PointLabel<P> | null;
  others: PointLabel<P>[];
}

export interface ScreenPoint {
  x: number;
  y: number;
}

export class PointLabelRanking {
  /**
   * Ranks the highlighted `points` once per change of the highlights (not per
   * frame): the selected point's label first, then the rest by similarity, ties
   * keeping their order. The selected point gets a label only if it is highlighted.
   * With a `measurer`, each text is measured here, so redraws don't measure it.
   */
  static rank<P extends LabelPoint>(
    points: readonly P[],
    similarities: ReadonlyMap<number, number>,
    selectedIndex: number | null | undefined,
    measurer?: TextMeasurer | null,
  ): RankedPointLabels<P> {
    if (measurer) measurer.font = LabelFonts.point();
    let selected: PointLabel<P> | null = null;
    const others: PointLabel<P>[] = [];
    for (const point of points) {
      const text = point.label || point.id;
      const label: PointLabel<P> = {
        point,
        text,
        width: measurer?.measureText(text).width,
        similarity: similarities.get(point.index) ?? 0,
        selected: point.index === selectedIndex,
      };
      if (label.selected) selected = label;
      else others.push(label);
    }
    others.sort((a, b) => b.similarity - a.similarity);
    return { selected, others };
  }
}

/**
 * Draws search labels to the right of their points, greedily: each label takes
 * its box if the grid has room, so earlier labels win. As in TensorBoard's
 * projector, a 1 px probe where the text would start is tested first, so labels
 * in crowded areas are rejected before their text is measured (Apache 2.0; notice
 * in THIRD_PARTY_NOTICES.md at the frontend root).
 */
export class PointLabelPass {
  constructor(private readonly style: LabelStyle) {}

  /** The text offset that clears a marker `diameter` px across: its radius plus a 4 px gap. */
  static offsetBeside(diameter: number): number {
    return diameter / 2 + 4;
  }

  /** Draws the labels that fit, `offset` px right of their projected points; returns their boxes. */
  draw<P>(
    ctx: LabelPaintContext,
    grid: CollisionGrid,
    labels: readonly PointLabel<P>[],
    project: (point: P) => ScreenPoint | null,
    offset: number,
  ): BoundingBox[] {
    const fontSize = LabelFonts.POINT_SIZE;
    ctx.font = LabelFonts.point(fontSize);
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.lineJoin = 'round';

    const drawn: BoundingBox[] = [];
    for (const label of labels) {
      const screen = project(label.point);
      if (!screen) continue;
      if (!grid.insert(LabelBox.probe(screen.x, screen.y, offset, fontSize), true)) continue;
      const width = label.width ?? ctx.measureText(label.text).width;
      const box = LabelBox.rightOf(screen.x, screen.y, offset, width, fontSize);
      if (!grid.insert(box)) continue;

      ctx.globalAlpha = LabelStyle.pointAlpha(label.similarity, label.selected);
      ctx.strokeStyle = this.style.pointStroke.color;
      ctx.lineWidth = this.style.pointStroke.width;
      ctx.strokeText(label.text, screen.x + offset, screen.y);
      ctx.fillStyle = this.style.pointFill;
      ctx.fillText(label.text, screen.x + offset, screen.y);
      drawn.push(box);
    }
    ctx.globalAlpha = 1;
    return drawn;
  }
}
