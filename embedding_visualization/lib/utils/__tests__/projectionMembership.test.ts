import { describe, expect, it } from 'vitest';

import { ProjectionMembership } from '../projectionMembership';

describe('ProjectionMembership', () => {
  it('accepts an exact ordered-item signature match', () => {
    expect(() => {
      ProjectionMembership.assertCompatible('abc123', 'abc123', 'umap_3d');
    }).not.toThrow();
  });

  it('rejects a projection whose ordered item membership differs from the core', () => {
    expect(() => {
      ProjectionMembership.assertCompatible('core-signature', 'other-signature', 'pca_2d');
    }).toThrow('pca_2d item membership does not match the loaded collection core');
  });

  it('rejects a projection when the core signature is missing', () => {
    expect(() => {
      ProjectionMembership.assertCompatible(null, 'projection-signature', 'umap_2d');
    }).toThrow('loaded collection core is missing its item membership signature');
  });

  it('rejects a projection response whose signature is missing', () => {
    expect(() => {
      ProjectionMembership.assertCompatible('core-signature', null, 'umap_2d');
    }).toThrow('umap_2d response is missing its item membership signature');
  });
});
