import { describe, expect, it } from 'vitest';
import { GraphExploreRoute, GraphDatasetLoader, CosmographDataAdapter, GraphNodeRepository } from '../graphs';

const dataset = {
  manifest: { schemaVersion: 1, graphId: 'cofire', revision: 'r1', title: 'Co-firing', nodeKind: 'sae_feature', directed: false, nodeCount: 2, edgeCount: 1, positions: 'none', linkedCollectionNames: ['features'], sae: { modelId: 'm', saeId: 's', width: 2 } },
  queryUrl: '/graphs/cofire/r1/query', pointsTable: 'graph_nodes', linksTable: 'graph_edges',
} as const;

describe('graph Explore routing', () => {
  it('recognizes graph-only routes before the catalog arrives', () => {
    expect(GraphExploreRoute.isGraph(new URLSearchParams('graph=cofire&view=graph'))).toBe(true);
    expect(GraphExploreRoute.isGraph(new URLSearchParams('collection=features'))).toBe(false);
  });
  it('preserves a linked collection and removes graph routing on scatter return', () => {
    const graph = GraphExploreRoute.graph('?collection=features&colorBy=density', 'cofire', 'features');
    expect(new URLSearchParams(graph).get('collection')).toBe('features');
    expect(new URLSearchParams(graph).get('graph')).toBe('cofire');
    const scatter = new URLSearchParams(GraphExploreRoute.scatter(graph, 'another'));
    expect(scatter.get('collection')).toBe('another');
    expect(scatter.has('graph')).toBe(false);
    expect(scatter.has('view')).toBe(false);
  });
  it('clears scatter-specific parameters for standalone graphs', () => {
    const url = new URLSearchParams(GraphExploreRoute.graph('?collection=features&preset=demo&colorBy=density', 'cofire'));
    expect(url.has('collection')).toBe(false);
    expect(url.has('preset')).toBe(false);
    expect(url.has('colorBy')).toBe(false);
  });
});

describe('remote graph data', () => {
  it('resolves deployment URLs and explicit indices without loading Arrow', () => {
    const resolved = GraphDatasetLoader.resolve(dataset, 'http://localhost:8000');
    expect(resolved.queryUrl).toBe('http://localhost:8000/graphs/cofire/r1/query');
    const config = CosmographDataAdapter.config(resolved);
    expect(config).toMatchObject({ points: 'graph_nodes', links: 'graph_edges', spaceDimensions: 3, pointIndexBy: 'index', linkSourceIndexBy: 'sourceIndex', linkTargetIndexBy: 'targetIndex', linkStrengthBy: 'weight' });
    expect(config.pointXBy).toBeUndefined();
  });
  it('maps unordered SQL results by index and suppresses old repository data', async () => {
    const rows = [{ index: 1, id: 'm::s::0', featureIndex: 0 }, { index: 0, id: 'm::s::1', featureIndex: 1 }];
    let resolve!: (value: unknown) => void;
    const api = { getPointsByIndices: () => new Promise(r => { resolve = r; }), convertCosmographDataToObject: (v: unknown) => v as typeof rows };
    const repository = new GraphNodeRepository(api);
    const pending = repository.nodes([0, 1]);
    resolve(rows);
    expect((await pending).map(n => n.featureIndex)).toEqual([1, 0]);
    const stale = repository.nodes([2]);
    repository.dispose();
    resolve([{ index: 2, id: 'stale' }]);
    expect(await stale).toEqual([]);
  });
});
