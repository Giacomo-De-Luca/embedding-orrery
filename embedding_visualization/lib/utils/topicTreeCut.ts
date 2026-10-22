/**
 * Merge trees over an extraction's base topics, and cutting them: the
 * browser's mirror of `TopicMergeTree` / `TopicTreeCut` in
 * interpretability_backend/backend/utils/topic_merge_tree.py. Both sides are
 * tested on the same literal examples (lib/utils/__tests__/topicTreeCut.test.ts
 * and unit_tests/test_topic_merge_tree.py), so the Explore slider's preview
 * and the saved reduction cannot drift apart.
 *
 * Layout (scipy's linkage convention): with k leaves, row i merges the nodes
 * `children[i]` at `heights[i]`; an index below k is a leaf (an index into
 * `leafIds`), k + j is the node made by row j. Rows are in merge order, so a
 * cut at N topics applies the first k − N rows. Leaves are base topics
 * (original clusters); noise is never a leaf.
 */
import type { TopicMergeTreeData, TopicTreeLinkage } from '../graphql/mutations';

export interface TopicMergeTreeInit {
  linkage: string;
  leafIds: readonly number[];
  leafSizes: readonly number[];
  children: readonly (readonly [number, number])[] | readonly (readonly number[])[];
  heights: readonly number[];
  /** Density trees only: noise item id → the merge row whose branch it fell out of. */
  noiseMerges?: ReadonlyMap<string, number>;
}

/** A validated dendrogram over base topics (see the module docstring). */
export class TopicMergeTree {
  readonly linkage: string;
  readonly leafIds: readonly number[];
  readonly leafSizes: readonly number[];
  readonly children: readonly (readonly [number, number])[];
  readonly heights: readonly number[];
  readonly noiseMerges: ReadonlyMap<string, number>;

  constructor(init: TopicMergeTreeInit) {
    this.linkage = init.linkage;
    this.leafIds = [...init.leafIds];
    this.leafSizes = [...init.leafSizes];
    this.children = init.children.map((pair) => [pair[0], pair[1]] as const);
    this.heights = [...init.heights];
    this.noiseMerges = new Map(init.noiseMerges ?? []);
    this.validate();
  }

  get nLeaves(): number {
    return this.leafIds.length;
  }

  /** Documents under every node: leaves first, then one entry per merge row. */
  nodeSizes(): number[] {
    const sizes = [...this.leafSizes];
    for (const [a, b] of this.children) sizes.push(sizes[a] + sizes[b]);
    return sizes;
  }

  /** The same tree with noise items mapped to merge rows. */
  withNoise(noiseMerges: ReadonlyMap<string, number>): TopicMergeTree {
    return new TopicMergeTree({ ...this, noiseMerges });
  }

  private validate(): void {
    const k = this.nLeaves;
    if (k === 0) throw new Error('A merge tree needs at least one leaf');
    if (this.leafSizes.length !== k) throw new Error(`Expected ${k} leaf sizes, got ${this.leafSizes.length}`);
    if (this.children.length !== k - 1 || this.heights.length !== k - 1) {
      throw new Error(`A tree over ${k} leaves needs ${k - 1} merge rows`);
    }
    const used = new Set<number>();
    this.children.forEach((pair, row) => {
      for (const node of pair) {
        if (!Number.isInteger(node) || node < 0 || node >= k + row) {
          throw new Error(`Merge row ${row} uses node ${node} before it exists`);
        }
        if (used.has(node)) throw new Error(`Node ${node} can be merged only once`);
        used.add(node);
      }
    });
    for (let row = 1; row < this.heights.length; row++) {
      if (this.heights[row] < this.heights[row - 1]) throw new Error('Merge heights must be non-decreasing');
    }
    for (const [item, row] of this.noiseMerges) {
      if (!Number.isInteger(row) || row < 0 || row >= k - 1) {
        throw new Error(`Noise item ${JSON.stringify(item)} maps to missing merge row ${row}`);
      }
    }
  }
}

/**
 * Groups of a merge tree after its first `nMerges` rows are applied. Groups
 * are named by their root node index (a leaf index for a topic not merged
 * yet), so a group keeps its name while it does not change.
 */
export class TopicTreeCut {
  constructor(readonly tree: TopicMergeTree) {}

  /** Rows to apply to leave `nTopics` groups. */
  mergesFor(nTopics: number): number {
    const k = this.tree.nLeaves;
    if (!Number.isInteger(nTopics) || nTopics < 1 || nTopics > k) {
      throw new Error(`Topic count must be between 1 and ${k}, got ${nTopics}`);
    }
    return k - nTopics;
  }

  /** Rows whose height is at most `height` (heights are sorted). */
  mergesWithin(height: number): number {
    const heights = this.tree.heights;
    let lo = 0;
    let hi = heights.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (heights[mid] <= height) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /** Root node of every node once the first `nMerges` rows are applied. */
  roots(nMerges: number): number[] {
    const k = this.tree.nLeaves;
    if (!Number.isInteger(nMerges) || nMerges < 0 || nMerges > k - 1) {
      throw new Error(`Merge count must be between 0 and ${k - 1}, got ${nMerges}`);
    }
    const parent = Array.from({ length: 2 * k - 1 }, (_, node) => node);
    for (let row = 0; row < nMerges; row++) {
      const [a, b] = this.tree.children[row];
      parent[a] = k + row;
      parent[b] = k + row;
    }
    // A parent's index is always larger than its children's, so resolving
    // from the top down leaves every node pointing at its final root.
    const roots = [...parent];
    for (let node = 2 * k - 2; node >= 0; node--) roots[node] = roots[parent[node]];
    return roots;
  }

  /** Group (root node) of each leaf. */
  groups(nMerges: number): number[] {
    return this.roots(nMerges).slice(0, this.tree.nLeaves);
  }

  /** The tree's noise items whose merge row has been applied, with the group they join. */
  absorbedNoise(nMerges: number): Map<string, number> {
    return this.absorbed(this.tree.noiseMerges, nMerges);
  }

  /** `absorbedNoise` for any noise keys mapped to merge rows. */
  absorbed<K>(noiseMerges: ReadonlyMap<K, number>, nMerges: number): Map<K, number> {
    const k = this.tree.nLeaves;
    const roots = this.roots(nMerges);
    const absorbed = new Map<K, number>();
    for (const [key, row] of noiseMerges) {
      if (row < nMerges) absorbed.set(key, roots[k + row]);
    }
    return absorbed;
  }

  /** Each group's largest leaf, as a base topic id (ties: the lowest id). */
  representatives(groups: readonly number[]): Map<number, number> {
    const best = new Map<number, LeafRef>();
    groups.forEach((group, leaf) => {
      const candidate = { topicId: this.tree.leafIds[leaf], size: this.tree.leafSizes[leaf] };
      const current = best.get(group);
      best.set(group, current ? TopicTreeCut.preferred(current, candidate) : candidate);
    });
    return new Map([...best].map(([group, { topicId }]) => [group, topicId]));
  }

  /** The representative of two leaves: the larger one, ties to the lowest topic id. */
  static preferred(a: LeafRef, b: LeafRef): LeafRef {
    if (a.size !== b.size) return a.size > b.size ? a : b;
    return a.topicId <= b.topicId ? a : b;
  }
}

/** A leaf as representative candidate. */
export interface LeafRef {
  topicId: number;
  size: number;
}

/** Base level of a tree: which item fields hold its leaves' topic ids and labels. */
export type TopicTreeBaseLevel = 'topic' | 'subtopic';

/**
 * A merge tree as the `topicMergeTree` query describes it: the tree, its
 * leaves' current labels and where items keep their base topic ids.
 */
export class TopicTreeView {
  constructor(
    readonly extractionId: string,
    readonly baseLevel: TopicTreeBaseLevel,
    readonly tree: TopicMergeTree,
    /** Base topic id → label (as the items' metadata carries it). */
    readonly labels: ReadonlyMap<number, string>,
    readonly noiseCount: number,
    readonly availableLinkages: readonly TopicTreeLinkage[],
  ) {}

  /** Item field holding the base topic id (a reduced extraction keeps them as subtopics). */
  get baseField(): 'topic_id' | 'subtopic_id' {
    return this.baseLevel === 'subtopic' ? 'subtopic_id' : 'topic_id';
  }

  /** Item field holding the base topic label. */
  get baseLabelField(): 'topic_label' | 'subtopic_label' {
    return this.baseLevel === 'subtopic' ? 'subtopic_label' : 'topic_label';
  }

  /** Parse (and validate) a `topicMergeTree` result. Throws on a malformed tree. */
  static fromGraphQL(data: TopicMergeTreeData): TopicTreeView {
    const noiseIds = data.noiseItemIds ?? [];
    const noiseRows = data.noiseMerges ?? [];
    if (noiseIds.length !== noiseRows.length) {
      throw new Error(`Expected one merge row per noise item, got ${noiseRows.length} for ${noiseIds.length}`);
    }
    const tree = new TopicMergeTree({
      linkage: data.linkage,
      leafIds: data.leaves.map((leaf) => leaf.topicId),
      leafSizes: data.leaves.map((leaf) => leaf.size),
      children: data.merges.map((merge) => [merge.left, merge.right] as const),
      heights: data.merges.map((merge) => merge.height),
      noiseMerges: new Map(noiseIds.map((id, i) => [id, noiseRows[i]])),
    });
    return new TopicTreeView(
      data.extractionId,
      data.baseLevel,
      tree,
      new Map(data.leaves.map((leaf) => [leaf.topicId, leaf.label])),
      data.noiseCount,
      data.availableLinkages,
    );
  }
}
