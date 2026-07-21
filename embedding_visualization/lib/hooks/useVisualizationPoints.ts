import { useMemo, useRef } from 'react';
import type {
  DimensionMode,
  EmbeddingData,
  ProjectionMethod,
} from '../types/types';
import {
  VisualizationPointBuilder,
  type VisualizationPointsResult,
} from '../utils/visualizationPointBuilder';

export function useVisualizationPoints(
  data: EmbeddingData | null | undefined,
  visualizationState: {
    method: ProjectionMethod;
    mode: DimensionMode;
    searchQuery?: string;
  },
): VisualizationPointsResult {
  const manualWarningShown = useRef(false);

  return useMemo(() => {
    if (!data) {
      return { points2d: [], points3d: [] };
    }

    if (visualizationState.method === 'manual') {
      if (!manualWarningShown.current) {
        console.warn(
          'Manual dimension selection requires raw embeddings - using PCA as fallback',
        );
        manualWarningShown.current = true;
      }
    }

    return VisualizationPointBuilder.build(
      data,
      visualizationState.method,
      visualizationState.mode,
    );
  }, [data, visualizationState.method, visualizationState.mode]);
}
