import type {
  DimensionMode,
  EmbeddingData,
  Point2D,
  Point3D,
  ProjectionMethod,
} from '../types/types';

export interface VisualizationPointsResult {
  points2d: Point2D[];
  points3d: Point3D[];
}

/** Builds point wrappers only for the active visualization dimension. */
export class VisualizationPointBuilder {
  private static metadataValue(
    metadata: Record<string, unknown>,
    fieldName: string | null,
    fallback: string,
  ): string {
    if (!fieldName) return fallback;
    const value = metadata[fieldName];
    if (value === null || value === undefined) return fallback;
    return String(value);
  }

  static build(
    data: EmbeddingData,
    method: ProjectionMethod,
    mode: DimensionMode,
  ): VisualizationPointsResult {
    const { ids, documents, itemMetadata, projections, displayConfig } = data;
    const useUmap = method === 'umap';

    if (mode === '2d') {
      const coordinates = useUmap ? projections.umap_2d : projections.pca_2d;
      if (!coordinates) return { points2d: [], points3d: [] };

      const points2d = coordinates.map((coordinate, index): Point2D => {
        const metadata = itemMetadata[index] || {};
        return {
          x: coordinate[0],
          y: coordinate[1],
          id: ids[index],
          label: this.metadataValue(metadata, displayConfig.labelField, ids[index]),
          document: documents[index] || '',
          category: this.metadataValue(metadata, displayConfig.categoryField, ''),
          index,
          metadata,
        };
      });
      return { points2d, points3d: [] };
    }

    const coordinates = useUmap ? projections.umap_3d : projections.pca_3d;
    if (!coordinates) return { points2d: [], points3d: [] };

    const points3d = coordinates.map((coordinate, index): Point3D => {
      const metadata = itemMetadata[index] || {};
      return {
        x: coordinate[0],
        y: coordinate[1],
        z: coordinate[2],
        id: ids[index],
        label: this.metadataValue(metadata, displayConfig.labelField, ids[index]),
        document: documents[index] || '',
        category: this.metadataValue(metadata, displayConfig.categoryField, ''),
        index,
        metadata,
      };
    });
    return { points2d: [], points3d };
  }
}
