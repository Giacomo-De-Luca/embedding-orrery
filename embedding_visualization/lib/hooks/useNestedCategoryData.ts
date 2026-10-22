import { useMemo } from 'react';
import type { MetadataPoint, NestedColorMap } from '../types/types';
import { isNestedColorAvailable, buildNestedColorMap } from '../utils/nestedColorUtils';
import { useVisualizationStore } from '../stores/useVisualizationStore';

interface NestedCategoryResult {
  available: boolean;
  nestedColorMap: NestedColorMap | null;
}

/**
 * Memoized hook for nested topic/subtopic color mapping.
 * Returns availability flag and the color map (only computed when mode is active).
 */
export function useNestedCategoryData(
  points: MetadataPoint[],
  colorByField: string | null | undefined,
  nestedColorMode: boolean | undefined,
  palette?: string
): NestedCategoryResult {
  const topicOverrides = useVisualizationStore(
    (s) => s.categoryColorOverrides['topic_label']
  );
  // Topic-tree preview: groups keep their representative topic's saved hue.
  const pin = useVisualizationStore((s) => s.categoryColorPins['topic_label']);
  const sourceOverrides = useVisualizationStore((s) =>
    pin ? s.categoryColorOverrides[pin.sourceField] : undefined,
  );

  const available = useMemo(
    () => isNestedColorAvailable(points, colorByField),
    [points, colorByField]
  );

  const nestedColorMap = useMemo(() => {
    if (!available || !nestedColorMode) return null;
    return buildNestedColorMap(points, palette, topicOverrides, pin ? { pin, sourceOverrides } : undefined);
  }, [available, nestedColorMode, points, palette, topicOverrides, pin, sourceOverrides]);

  return { available, nestedColorMap };
}
