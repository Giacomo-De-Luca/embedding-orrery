import type { CollisionGrid } from '../collisionGrid';
import type { LabelPaintContext } from './labelCanvas';
import { LabelBox } from './labelBox';
import type { LabelHitBoxes } from './labelHitBoxes';
import { LabelFonts, type LabelStyle } from './labelStyle';

/** A cluster label ready to draw at a screen position. */
export interface ClusterLabelDraw {
  text: string;
  /** Screen position of the text's centre. */
  x: number;
  y: number;
  /** The text's width at its font, when already measured; measured on the canvas otherwise. */
  width?: number;
  fontSize: number;
  /** The cluster colour, before `LabelStyle.clusterFill` desaturates it. */
  color: string;
  alpha: number;
  /** What a click on the label reports, or null when it isn't clickable. */
  hitLabel: string | null;
}

/** The order cluster labels claim space in: earlier labels win collisions. */
export class ClusterLabelOrder {
  /**
   * 3D: the clusters nearest the camera first (ties: more points first), capped at
   * max(8, ⌊2.5·√n⌋) and fading from 1 to 0.65 with distance. With "show all",
   * every cluster in its own order, fully opaque.
   */
  static nearest<L extends { pointCount: number }>(
    labels: readonly L[],
    distanceOf: (label: L) => number,
    showAll: boolean,
  ): { label: L; alpha: number }[] {
    if (showAll) return labels.map(label => ({ label, alpha: 1 }));

    const sorted = labels
      .map(label => ({ label, distance: distanceOf(label) }))
      .sort((a, b) => a.distance - b.distance || b.label.pointCount - a.label.pointCount);
    const count = Math.min(sorted.length, Math.max(8, Math.floor(Math.sqrt(sorted.length) * 2.5)));
    const chosen = sorted.slice(0, count);
    const nearest = chosen[0]?.distance ?? 0;
    const span = (chosen[chosen.length - 1]?.distance ?? 1) - nearest || 1;
    return chosen.map(({ label, distance }) => ({ label, alpha: 1 - 0.35 * ((distance - nearest) / span) }));
  }

  /** 2D: the highest priority first (ties keep their order), so a bigger cluster's label wins. */
  static byPriority<L extends { priority: number }>(labels: readonly L[]): L[] {
    return [...labels].sort((a, b) => b.priority - a.priority);
  }
}

/** Draws cluster labels centred on their positions through a collision grid. */
export class ClusterLabelPainter {
  constructor(private readonly style: LabelStyle) {}

  /** Draws each label whose box is free, in order, and records the clickable ones in `hits`. */
  paint(ctx: LabelPaintContext, grid: CollisionGrid, labels: Iterable<ClusterLabelDraw>, hits: LabelHitBoxes): void {
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    ctx.lineJoin = 'round';
    for (const label of labels) {
      ctx.font = LabelFonts.cluster(label.fontSize);
      const width = label.width ?? ctx.measureText(label.text).width;
      const box = LabelBox.centered(label.x, label.y, width, label.fontSize);
      if (!grid.insert(box)) continue;

      ctx.globalAlpha = label.alpha;
      ctx.strokeStyle = this.style.clusterStroke.color;
      ctx.lineWidth = this.style.clusterStroke.width;
      ctx.strokeText(label.text, label.x, label.y);
      ctx.fillStyle = this.style.clusterFill(label.color);
      ctx.fillText(label.text, label.x, label.y);
      if (label.hitLabel !== null) hits.add(box, label.hitLabel);
    }
    ctx.globalAlpha = 1;
  }
}
