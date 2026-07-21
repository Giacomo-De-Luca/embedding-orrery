import { describe, expect, it } from 'vitest';

import { ProjectionLoadPolicy } from '../projectionLoadPolicy';

describe('ProjectionLoadPolicy', () => {
  it('requests only the active UMAP dimension for a new 3D collection', () => {
    expect(ProjectionLoadPolicy.forView('umap', '3d', true)).toEqual({
      projectionTypes: ['umap_3d'],
      includeCore: true,
    });
  });

  it('requests only the active PCA dimension for a follow-up 2D load', () => {
    expect(ProjectionLoadPolicy.forView('pca', '2d', false)).toEqual({
      projectionTypes: ['pca_2d'],
      includeCore: false,
    });
  });

  it('keeps the manual-mode PCA fallback dimension-aware', () => {
    expect(ProjectionLoadPolicy.forView('manual', '3d', false)).toEqual({
      projectionTypes: ['pca_3d'],
      includeCore: false,
    });
  });
});
