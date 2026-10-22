/**
 * The topic fields items show while the Explore topic slider previews a cut
 * of the collection's merge tree (see topicTreeCut.ts).
 *
 * `TopicTreeGroups` names every node of a tree once (O(base topics)): a cut's
 * groups are tree nodes, so every label any cut can show is known up front,
 * which keeps the colour pin unchanged while the slider moves. A cut
 * (`TopicTreeGroups.cut`) is cheap; `TopicCutPreview.apply` rewrites the item
 * metadata (O(items)) and is the part callers defer. While previewing, every
 * item with a base topic shows:
 * - `topic_id`: its group's representative base topic id (the largest member,
 *   so ids stay stable while sliding) and `topic_label`: the representative's
 *   label plus " +m" for the m other members;
 * - `subtopic_id` / `subtopic_label`: its own base topic, so "Color by
 *   subtopics" works in the preview even for an unreduced extraction.
 * Noise (base id -1) is noise in the preview unless a density cut absorbs it,
 * in which case it joins the group with subtopic "-1" / "Unclustered", as a
 * saved density reduction stores it.
 */
import { CategoryKey } from './categoryKey';
import { TopicTreeCut, type LeafRef, type TopicTreeView } from './topicTreeCut';

/** A topic of the preview: the base topics under one tree node. */
export interface TopicCutGroup {
  /** Representative base topic id, as the item field carries it. */
  topicId: string;
  label: string;
  /** Base topics in the group. */
  members: number;
  /** The representative's own label (its colour source). */
  representativeLabel: string;
}

type ItemMetadata = Record<string, unknown>;

/** Every node of a merge tree as a preview group (representative, label, size). */
export class TopicTreeGroups {
  private readonly nodes: TopicCutGroup[];
  private readonly baseLabels: ReadonlyMap<string, string>;

  constructor(readonly view: TopicTreeView) {
    const { tree } = view;
    const best: LeafRef[] = tree.leafIds.map((topicId, leaf) => ({ topicId, size: tree.leafSizes[leaf] }));
    const members: number[] = tree.leafIds.map(() => 1);
    for (const [a, b] of tree.children) {
      best.push(TopicTreeCut.preferred(best[a], best[b]));
      members.push(members[a] + members[b]);
    }
    this.nodes = best.map(({ topicId }, node) => {
      const representativeLabel = this.labelOf(topicId);
      const count = members[node];
      return {
        topicId: String(topicId),
        label: count > 1 ? `${representativeLabel} +${count - 1}` : representativeLabel,
        members: count,
        representativeLabel,
      };
    });
    this.baseLabels = new Map(tree.leafIds.map((topicId) => [String(topicId), this.labelOf(topicId)]));
  }

  /** The group a cut shows for tree node `node`. */
  node(node: number): TopicCutGroup {
    return this.nodes[node];
  }

  /** Label of a base topic (by its id as items carry it). */
  baseLabel(baseId: string): string | undefined {
    return this.baseLabels.get(baseId);
  }

  /**
   * Every merged group label any cut of this tree can show → its
   * representative's label: the value map of the preview's colour pin.
   */
  representativeLabels(): Record<string, string> {
    const map: Record<string, string> = {};
    for (const group of this.nodes) {
      if (group.members > 1) map[group.label] = group.representativeLabel;
    }
    return map;
  }

  /** The preview at `nTopics` topics; density noise is absorbed only with `absorbNoise`. */
  cut(nTopics: number, absorbNoise: boolean): TopicCutPreview {
    return new TopicCutPreview(this, nTopics, absorbNoise);
  }

  private labelOf(topicId: number): string {
    return this.view.labels.get(topicId) || `Topic ${topicId}`;
  }
}

/** One cut of a tree: its groups, and the item metadata it shows. */
export class TopicCutPreview {
  static readonly NOISE_ID = '-1';
  static readonly NOISE_LABEL = 'Unclustered';
  /**
   * Collections up to this size recolour while the slider is dragged; larger
   * ones on release (every step rewrites the metadata and redraws the plot).
   */
  static readonly LIVE_SLIDER_MAX_ITEMS = 50_000;

  /** Groups by root node, in leaf order. */
  readonly groups: ReadonlyMap<number, TopicCutGroup>;
  private readonly groupOfBase: ReadonlyMap<string, TopicCutGroup>;
  private readonly absorbed: ReadonlyMap<string, TopicCutGroup>;

  constructor(readonly treeGroups: TopicTreeGroups, readonly nTopics: number, absorbNoise: boolean) {
    const { tree } = treeGroups.view;
    const cut = new TopicTreeCut(tree);
    const merges = cut.mergesFor(nTopics);
    const groups = new Map<number, TopicCutGroup>();
    const groupOfBase = new Map<string, TopicCutGroup>();
    cut.groups(merges).forEach((root, leaf) => {
      const group = treeGroups.node(root);
      groups.set(root, group);
      groupOfBase.set(String(tree.leafIds[leaf]), group);
    });
    const absorbed = new Map<string, TopicCutGroup>();
    if (absorbNoise) {
      for (const [itemId, root] of cut.absorbedNoise(merges)) absorbed.set(itemId, treeGroups.node(root));
    }
    this.groups = groups;
    this.groupOfBase = groupOfBase;
    this.absorbed = absorbed;
  }

  /** Number of topics in the preview (noise not counted). */
  get size(): number {
    return this.groups.size;
  }

  /**
   * `itemMetadata` (aligned with `ids`) with the preview's topic fields. Returns
   * a new array; items whose four topic fields do not change keep their object,
   * changed items are shallow copies. Items without the base field, or whose
   * base topic the tree does not know (a stale tree), stay untouched.
   */
  apply(itemMetadata: readonly ItemMetadata[], ids: readonly string[]): ItemMetadata[] {
    const baseField = this.treeGroups.view.baseField;
    const result = new Array<ItemMetadata>(itemMetadata.length);
    for (let i = 0; i < itemMetadata.length; i++) {
      const meta = itemMetadata[i];
      const raw = meta?.[baseField];
      if (CategoryKey.isMissing(raw)) {
        result[i] = meta;
        continue;
      }
      const baseId = String(raw);
      let topicId: string;
      let topicLabel: string;
      let subtopicId: string;
      let subtopicLabel: string;
      if (baseId === TopicCutPreview.NOISE_ID) {
        const group = this.absorbed.get(ids[i]);
        topicId = group?.topicId ?? TopicCutPreview.NOISE_ID;
        topicLabel = group?.label ?? TopicCutPreview.NOISE_LABEL;
        subtopicId = TopicCutPreview.NOISE_ID;
        subtopicLabel = TopicCutPreview.NOISE_LABEL;
      } else {
        const group = this.groupOfBase.get(baseId);
        if (!group) {
          result[i] = meta;
          continue;
        }
        topicId = group.topicId;
        topicLabel = group.label;
        subtopicId = baseId;
        subtopicLabel = this.treeGroups.baseLabel(baseId)!;
      }
      result[i] =
        meta.topic_id === topicId &&
        meta.topic_label === topicLabel &&
        meta.subtopic_id === subtopicId &&
        meta.subtopic_label === subtopicLabel
          ? meta
          : { ...meta, topic_id: topicId, topic_label: topicLabel, subtopic_id: subtopicId, subtopic_label: subtopicLabel };
    }
    return result;
  }
}
