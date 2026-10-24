/** Looking up the points a search highlighted. */
export class HighlightedPoints {
  /**
   * The highlighted points in highlight order. O(k) for k highlights when
   * `points[i].index === i`, as `VisualizationPointBuilder` builds them; a single
   * scan otherwise.
   */
  static lookup<P extends { index: number }>(
    points: readonly P[],
    highlights: ReadonlyMap<number, unknown> | null | undefined,
  ): P[] {
    if (!highlights || highlights.size === 0) return [];
    const result: P[] = [];
    if (HighlightedPoints.isIndexAligned(points)) {
      for (const index of highlights.keys()) {
        const point = points[index];
        if (point !== undefined && point.index === index) result.push(point);
      }
      return result;
    }
    const byIndex = new Map(points.map(point => [point.index, point]));
    for (const index of highlights.keys()) {
      const point = byIndex.get(index);
      if (point !== undefined) result.push(point);
    }
    return result;
  }

  private static isIndexAligned(points: readonly { index: number }[]): boolean {
    const last = points.length - 1;
    return last < 0 || (points[0].index === 0 && points[last].index === last);
  }
}
