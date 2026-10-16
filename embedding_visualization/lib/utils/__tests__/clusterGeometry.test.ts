import { describe, expect, it } from 'vitest';
import { groupPointsByCluster } from '../clusterGeometry';
import type { Point3D } from '../../types/types';

const point = (index: number, x: number, metadata: Record<string, unknown>): Point3D => ({
  x, y: 0, z: 0, id: String(index), label: '', document: '', category: '', index, metadata,
});

describe('groupPointsByCluster', () => {
  it('builds clusters for real categories but not for missing values or noise', () => {
    const points = [
      point(0, 0, { category: 'a' }), point(1, 1, { category: 'a' }), point(2, 2, { category: 'a' }),
      point(3, 5, {}), point(4, 6, { category: null }), point(5, 7, { category: '' }),
      point(6, 8, { category: 'Unclustered' }), point(7, 9, { category: 'Unclustered' }), point(8, 9, { category: 'Unclustered' }),
    ];
    const clusters = groupPointsByCluster(points, 'category', { a: '#123456' });
    expect([...clusters.keys()]).toEqual(['a']);
    expect(clusters.get('a')?.centroid.x).toBe(1);
    expect(clusters.get('a')?.color).toBe('#123456');
  });
});
