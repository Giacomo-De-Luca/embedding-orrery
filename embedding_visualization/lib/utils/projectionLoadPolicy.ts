import type { DimensionMode, ProjectionMethod } from '../types/types';

export type ProjectionType = 'pca_2d' | 'pca_3d' | 'umap_2d' | 'umap_3d';

export interface ProjectionLoadRequest {
  projectionTypes: ProjectionType[];
  includeCore: boolean;
}

/**
 * Builds the minimal collection request for the active visualization.
 *
 * Core item data is needed once per collection. Later method/dimension changes
 * request only the missing projection so IDs, documents, and metadata are not
 * retransmitted and reparsed.
 */
export class ProjectionLoadPolicy {
  static forView(
    method: ProjectionMethod,
    mode: DimensionMode,
    isNewCollection: boolean,
  ): ProjectionLoadRequest {
    const prefix = method === 'umap' ? 'umap' : 'pca';
    return {
      projectionTypes: [`${prefix}_${mode}` as ProjectionType],
      includeCore: isNewCollection,
    };
  }
}
