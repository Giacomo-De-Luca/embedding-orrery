import type { BoundingBox } from '../collisionGrid';

/** The boxes of the clickable labels drawn last, for hit-testing clicks on the overlay. */
export class LabelHitBoxes {
  private readonly boxes: { box: BoundingBox; label: string }[] = [];

  get size(): number {
    return this.boxes.length;
  }

  add(box: BoundingBox, label: string): void {
    this.boxes.push({ box, label });
  }

  clear(): void {
    this.boxes.length = 0;
  }

  /** The labels whose box holds (x, y), edges included, in drawing order. */
  at(x: number, y: number): string[] {
    return this.boxes
      .filter(({ box }) => x >= box.loX && x <= box.hiX && y >= box.loY && y <= box.hiY)
      .map(({ label }) => label);
  }
}

/** What a click on a cluster label can do. */
export interface ClusterLabelClickHandlers {
  /** The topic id a label names, if any. */
  topicIdOf?: (label: string) => number | undefined;
  /** Toggles a topic. */
  onTopic?: (topicId: number) => void;
  /** Toggles a category (shift: add to the selection); returns whether the label was one. */
  onCategory?: (label: string, shiftKey: boolean) => boolean;
}

export class ClusterLabelClicks {
  /** A press that moves further than this before its release is a drag (pan, rotate). */
  static readonly DRAG_DISTANCE = 5;

  /** Whether a release `release` px from its press `press` ended a drag rather than a click. */
  static isDrag(press: { x: number; y: number } | null, release: { x: number; y: number }): boolean {
    if (!press) return false;
    const dx = release.x - press.x;
    const dy = release.y - press.y;
    return dx * dx + dy * dy > ClusterLabelClicks.DRAG_DISTANCE ** 2;
  }

  /** Whether a click on a cluster label can do anything. */
  static isClickable(handlers: ClusterLabelClickHandlers): boolean {
    return Boolean(handlers.onCategory || (handlers.onTopic && handlers.topicIdOf));
  }

  /**
   * Routes a click on `labels` (from `LabelHitBoxes.at`): a label naming a topic
   * toggles it, otherwise the category handler may take it; the next label is
   * tried when neither does. Returns whether a label took the click.
   */
  static route(labels: readonly string[], handlers: ClusterLabelClickHandlers, shiftKey: boolean): boolean {
    for (const label of labels) {
      const topicId = handlers.topicIdOf?.(label);
      if (handlers.onTopic && topicId !== undefined) {
        handlers.onTopic(topicId);
        return true;
      }
      if (handlers.onCategory?.(label, shiftKey)) return true;
    }
    return false;
  }
}
