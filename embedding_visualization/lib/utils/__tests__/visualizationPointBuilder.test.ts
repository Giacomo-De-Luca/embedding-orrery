import { describe, expect, it } from 'vitest';

import type { EmbeddingData } from '../../types/types';
import { VisualizationPointBuilder } from '../visualizationPointBuilder';

const data = (): EmbeddingData => ({
  ids: ['a', 'b'],
  documents: ['document a', 'document b'],
  itemMetadata: [{ label: 'Alpha' }, { label: 'Beta' }],
  availableFields: ['label'],
  displayConfig: {
    labelField: 'label',
    categoryField: null,
    categoryValues: [],
    categoryName: 'Category',
  },
  projections: {
    pca_2d: null,
    pca_3d: null,
    umap_2d: [[1, 2], [3, 4]],
    umap_3d: [[1, 2, 3], [4, 5, 6]],
  },
  metadata: {
    total_items: 2,
    embedding_dim: 3,
    timestamp: '',
  },
});

describe('VisualizationPointBuilder', () => {
  it('builds only 3D wrappers in 3D mode without requiring a 2D projection', () => {
    const input = data();
    input.projections.umap_2d = null;

    const result = VisualizationPointBuilder.build(input, 'umap', '3d');

    expect(result.points2d).toEqual([]);
    expect(result.points3d).toHaveLength(2);
    expect(result.points3d[0]).toMatchObject({
      x: 1,
      y: 2,
      z: 3,
      index: 0,
      id: 'a',
      label: 'Alpha',
    });
  });

  it('builds only 2D wrappers in 2D mode without requiring a 3D projection', () => {
    const input = data();
    input.projections.umap_3d = null;

    const result = VisualizationPointBuilder.build(input, 'umap', '2d');

    expect(result.points3d).toEqual([]);
    expect(result.points2d).toHaveLength(2);
    expect(result.points2d[1]).toMatchObject({
      x: 3,
      y: 4,
      index: 1,
      id: 'b',
      label: 'Beta',
    });
  });

  it('returns empty arrays when the active projection is unavailable', () => {
    const input = data();
    input.projections.pca_3d = null;

    expect(VisualizationPointBuilder.build(input, 'pca', '3d')).toEqual({
      points2d: [],
      points3d: [],
    });
  });
});
