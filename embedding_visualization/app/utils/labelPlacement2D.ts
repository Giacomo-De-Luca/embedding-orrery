/**
 * 2D cluster-label placement with Apple's dynamic map labeling
 * (`dynamicMapPlacement.ts`). Each label is laid out once at the initial view and
 * gets the zoom band in which it shows without overlapping a higher-priority
 * label, so labels appear and disappear smoothly as the user zooms, with larger
 * clusters visible further out, instead of flickering while panning.
 */

import { dynamicLabelPlacement, type Label, type Placement } from './dynamicMapPlacement';
import type { ClusterData } from '../../lib/utils/clusterGeometry';
import type { AxisRanges, CartesianView } from '../../lib/utils/cartesianView';
import { ClusterLabelOrder, type ClusterLabelDraw } from '../../lib/utils/labels/clusterLabels';
import { LabelBox } from '../../lib/utils/labels/labelBox';
import { LabelFonts, type TextMeasurer } from '../../lib/utils/labels/labelStyle';

export interface ClusterLabelPlacement {
  label: string;
  color: string;
  /** Data-space centroid */
  dataX: number;
  dataY: number;
  /** Text width in CSS pixels (at rendering font) */
  textWidth: number;
  /** Apple placement result (null = never visible) */
  placement: Placement | null;
  /** Priority (cluster point count) */
  priority: number;
  /** Font size in px; renderers default to `LabelFonts.CLUSTER_SIZE` when absent */
  fontSize?: number;
  /** Topic label backing this text, when it differs from `label` semantics
   *  (density labels use it to keep topic-click behavior) */
  topicLabel?: string | null;
}

/** One label to place: its text, data-space anchor, priority and font size. */
export interface ZoomBandEntry {
  label: string;
  dataX: number;
  dataY: number;
  priority: number;
  fontSize: number;
}

/** An entry with its measured text width and its zoom band (null = never visible). */
export type ZoomBandPlaced<E extends ZoomBandEntry> = E & { textWidth: number; placement: Placement | null };

/** The zoom-band placement both the centroid and the density cluster labels go through. */
export class ZoomBandPlacement {
  /** The furthest zoom-out (span ratio to the initial view) a band is computed for. */
  static readonly GLOBAL_MAX_SCALE = 2;
  /** Labels are drawn while their anchor is within this many px of the plot area. */
  static readonly CLIP_MARGIN = 20;

  /**
   * The cluster labels to draw in `view`: those whose zoom band holds its scale
   * relative to `initial`, or every label with "show all" (the collision grid then
   * declutters them), near the plot area. Runs every pan/zoom frame, so it keeps
   * the order it is given: `place` already returns placements by priority.
   */
  static visibleAt(
    placements: readonly ClusterLabelPlacement[],
    view: CartesianView,
    initial: AxisRanges,
    showAll: boolean,
  ): ClusterLabelDraw[] {
    const scale = view.scaleRelativeTo(initial);
    const draws: ClusterLabelDraw[] = [];
    for (const label of placements) {
      const band = label.placement;
      if (!showAll && (!band || scale < band.minScale || scale > band.maxScale)) continue;
      const screen = view.project(label.dataX, label.dataY);
      if (!view.contains(screen, ZoomBandPlacement.CLIP_MARGIN)) continue;
      draws.push({
        text: label.label,
        x: screen.x,
        y: screen.y,
        width: label.textWidth,
        fontSize: label.fontSize ?? LabelFonts.CLUSTER_SIZE,
        color: label.color,
        alpha: 1,
        // Explicit null disables topic clicks for other fields and summaries,
        // even when their text happens to match a topic name.
        hitLabel: label.topicLabel === undefined ? label.label : label.topicLabel,
      });
    }
    return draws;
  }

  /** The placement algorithm's label for an entry: its centred box at the initial view. */
  static labelFor(entry: ZoomBandEntry, textWidth: number, initial: CartesianView): Label {
    const anchor = initial.project(entry.dataX, entry.dataY);
    const box = LabelBox.centered(anchor.x, anchor.y, textWidth, entry.fontSize);
    return {
      bounds: { xMin: box.loX, yMin: box.loY, xMax: box.hiX, yMax: box.hiY },
      locationAtZero: anchor,
      priority: entry.priority,
    };
  }

  /**
   * Measures every entry in the cluster font at its size and computes its zoom
   * band; returns them highest priority first (ties in their order), the order
   * they then claim space in.
   */
  static place<E extends ZoomBandEntry>(entries: E[], ctx: TextMeasurer, initial: CartesianView): ZoomBandPlaced<E>[] {
    if (entries.length === 0) return [];
    const widths = entries.map(entry => {
      ctx.font = LabelFonts.cluster(entry.fontSize);
      return ctx.measureText(entry.label).width;
    });
    const placements = dynamicLabelPlacement(
      entries.map((entry, i) => ZoomBandPlacement.labelFor(entry, widths[i], initial)),
      { globalMaxScale: ZoomBandPlacement.GLOBAL_MAX_SCALE },
    );
    return ClusterLabelOrder.byPriority(
      entries.map((entry, i) => ({ ...entry, textWidth: widths[i], placement: placements[i] })),
    );
  }

  /**
   * One label per cluster at its centroid, prioritised by point count, at the
   * default cluster font size.
   */
  static forClusters(
    clusterDataMap: Map<string, ClusterData>,
    ctx: TextMeasurer,
    initial: CartesianView,
  ): ClusterLabelPlacement[] {
    const entries = Array.from(clusterDataMap, ([name, cluster]) => ({
      label: name,
      color: cluster.color,
      dataX: cluster.centroid.x,
      dataY: cluster.centroid.y,
      priority: cluster.points.length,
      fontSize: LabelFonts.CLUSTER_SIZE,
    }));
    return ZoomBandPlacement.place(entries, ctx, initial);
  }
}
