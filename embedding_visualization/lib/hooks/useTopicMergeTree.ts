'use client';

import { useCallback, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useApolloClient, useMutation, useQuery } from '@apollo/client/react';
import { toast } from 'sonner';

import { REDUCE_TOPICS, TOPIC_MERGE_TREE } from '../graphql/queries';
import type { ReduceTopicsResult, TopicMergeTreeData, TopicTreeLinkage } from '../graphql/mutations';
import { useVisualizationStore } from '../stores/useVisualizationStore';
import type { EmbeddingData } from '../types/types';
import type { CategoryColorPin } from '../utils/categoryColors';
import { CategoryKey } from '../utils/categoryKey';
import { announceJobOutcome } from '../utils/jobCancellation';
import { JobIds } from '../utils/jobIds';
import { TopicCutPreview, TopicTreeGroups } from '../utils/topicCutMetadata';
import { TopicTreeView } from '../utils/topicTreeCut';
import { DEFAULT_DISTANCE_THRESHOLD, ReductionInputs } from '../utils/topicReduction';
import { refreshTopicSimilarityOrders } from './useTopicSimilarityOrder';

interface TopicMergeTreeQueryData {
  topicMergeTree: TopicMergeTreeData | null;
}

type ColorPins = Record<string, CategoryColorPin>;

const NO_PINS: ColorPins = {};
const MIN_TOPICS = 2;

export interface UseTopicMergeTreeOptions {
  /** Fetch the tree and allow a preview: the plot is coloured by topic_label. */
  enabled: boolean;
  /** Called when a preview starts (the page clears the topic selection and filters). */
  onPreviewStart?: () => void;
  /** After a successful Apply: reload everything derived from the old topics. */
  onApplied?: () => Promise<unknown>;
}

export interface UseTopicMergeTreeReturn {
  /** Merge tree of the collection's active extraction; null while loading or unavailable. */
  view: TopicTreeView | null;
  loading: boolean;
  error: string | null;
  linkage: TopicTreeLinkage;
  setLinkage: (linkage: TopicTreeLinkage) => void;
  absorbNoise: boolean;
  setAbsorbNoise: (absorb: boolean) => void;
  minTopics: number;
  /** Base topics (the slider's maximum). */
  maxTopics: number;
  /** Topics the slider shows: the preview's count, else the saved count. */
  topicCount: number;
  savedTopicCount: number;
  /** Move the slider: starts (or moves) the preview. */
  setTopicCount: (nTopics: number) => void;
  previewing: boolean;
  /** Recolour while dragging; large collections recolour on release only. */
  liveSlider: boolean;
  /** Leave the preview (back to the saved topics). */
  reset: () => void;
  /** The collection data with the preview's topic fields; null when not previewing. */
  previewData: EmbeddingData | null;
  /** Save the previewed cut (`reduceTopics`); merged topics LLM-labelled on request. */
  apply: (regenerateLabels: boolean) => Promise<void>;
  applying: boolean;
  /** Why the last Apply did not save (a cancel or a failure); cleared by the next one. */
  applyError: string | null;
  /** Progress job of Apply (JobIds.topicReduction). */
  jobId: string | null;
}

/**
 * The Explore topic slider: fetches the merge tree over the active
 * extraction's base topics, keeps the cut (store `topicCut`), previews it by
 * rewriting the topic fields in the browser and pins the preview's colours,
 * and saves it with Apply. Feed `previewData ?? data` to useVisualizationPoints;
 * `data` here is the collection data after probes (`probes.augmentedData ?? data`).
 */
export function useTopicMergeTree(
  collectionName: string | null,
  data: EmbeddingData | null,
  options: UseTopicMergeTreeOptions,
): UseTopicMergeTreeReturn {
  const { enabled } = options;
  const client = useApolloClient();
  const cut = useVisualizationStore((s) => s.topicCut);
  const setTopicCut = useVisualizationStore((s) => s.setTopicCut);
  const setCategoryColorPins = useVisualizationStore((s) => s.setCategoryColorPins);

  const isDensity = cut.linkage === 'density';
  const {
    data: treeData,
    loading,
    error: queryError,
    refetch,
  } = useQuery<TopicMergeTreeQueryData>(TOPIC_MERGE_TREE, {
    variables: { collectionName: collectionName ?? '', linkage: cut.linkage, includeNoise: isDensity },
    skip: !enabled || !collectionName,
    fetchPolicy: 'network-only',
  });

  // While another linkage loads, keep cutting this collection's previous tree
  // (no flash of the saved topics in the middle of a preview). Tagged with its
  // collection: another collection's tree must never be cut on this one's items.
  const fresh = treeData?.topicMergeTree ?? null;
  const [kept, setKept] = useState<{ collection: string; tree: TopicMergeTreeData } | null>(null);
  if (fresh && collectionName && (kept?.tree !== fresh || kept.collection !== collectionName)) {
    setKept({ collection: collectionName, tree: fresh });
  }
  const raw = fresh ?? (loading && kept?.collection === collectionName ? kept.tree : null);
  const parsed = useMemo(() => {
    if (!raw) return { view: null, error: null };
    try {
      return { view: TopicTreeView.fromGraphQL(raw), error: null };
    } catch (err) {
      return { view: null, error: err instanceof Error ? err.message : 'Invalid topic tree' };
    }
  }, [raw]);
  const view = enabled ? parsed.view : null;

  // A linkage this collection does not offer (density without a density tree)
  // returns null: fall back to average rather than hiding the controls.
  const unavailable = enabled && !loading && !queryError && treeData !== undefined && treeData.topicMergeTree === null;
  useEffect(() => {
    if (unavailable && cut.linkage !== 'average') setTopicCut({ linkage: 'average' });
  }, [unavailable, cut.linkage, setTopicCut]);

  const savedTopicCount = useMemo(() => {
    const ids = new Set<string>();
    for (const meta of data?.itemMetadata ?? []) {
      const id = meta?.topic_id;
      if (!CategoryKey.isMissing(id) && String(id) !== TopicCutPreview.NOISE_ID) ids.add(String(id));
    }
    return ids.size;
  }, [data]);

  const maxTopics = view?.tree.nLeaves ?? 0;
  const clampTopics = (n: number) => Math.min(Math.max(Math.round(n), MIN_TOPICS), Math.max(maxTopics, MIN_TOPICS));
  const topicCount = clampTopics(cut.nTopics ?? savedTopicCount);
  const previewing = enabled && cut.previewing && view !== null && maxTopics >= MIN_TOPICS;
  const absorbNoise = isDensity && cut.absorbNoise;

  // The cut is cheap; rewriting every item's metadata is not, so it follows
  // the slider as a deferred value.
  const treeGroups = useMemo(() => (view ? new TopicTreeGroups(view) : null), [view]);
  const preview = useMemo(
    () => (previewing && treeGroups ? treeGroups.cut(topicCount, absorbNoise) : null),
    [previewing, treeGroups, topicCount, absorbNoise],
  );
  const deferredPreview = useDeferredValue(preview);
  const previewMetadata = useMemo(
    () => (deferredPreview && data ? deferredPreview.apply(data.itemMetadata, data.ids) : null),
    [deferredPreview, data],
  );

  // Colour pins: every label any cut of this tree shows → its representative,
  // coloured as that base topic is in the saved base field. Fixed per tree, so
  // moving the slider never republishes them. Switched on with the first
  // preview metadata, not before: the saved labels of a reduced extraction are
  // not in the pin's domain and would briefly take fresh palette slots.
  const baseLabelField = view?.baseLabelField ?? null;
  const domain = useMemo(
    () => (previewing && data && baseLabelField ? CategoryKey.valuesOf(data.itemMetadata, baseLabelField) : null),
    [previewing, data, baseLabelField],
  );
  const hasPreviewMetadata = previewMetadata !== null;
  const livePins = useMemo<ColorPins>(
    () =>
      previewing && hasPreviewMetadata && treeGroups && domain && baseLabelField
        ? { topic_label: { sourceField: baseLabelField, domain, valueMap: treeGroups.representativeLabels() } }
        : NO_PINS,
    [previewing, hasPreviewMetadata, treeGroups, domain, baseLabelField],
  );
  const livePreviewData = useMemo(
    () => (previewing && previewMetadata && data ? { ...data, itemMetadata: previewMetadata } : null),
    [previewing, previewMetadata, data],
  );

  // After a successful Apply the preview stays on screen, frozen, until the
  // saved topics have reloaded (the reloaded data no longer fits the old tree).
  // Tagged with its collection, so switching collections meanwhile drops it.
  const [frozenState, setFrozen] = useState<
    { collection: string; data: EmbeddingData | null; pins: ColorPins } | null
  >(null);
  const frozen = frozenState?.collection === collectionName ? frozenState : null;
  const previewData = frozen ? frozen.data : livePreviewData;
  const pins = frozen ? frozen.pins : livePins;

  // Published before paint, like useProbes' colour aliases.
  useLayoutEffect(() => {
    setCategoryColorPins(pins);
  }, [pins, setCategoryColorPins]);
  useEffect(() => () => setCategoryColorPins(NO_PINS), [setCategoryColorPins]);

  // Latest values for the async Apply (written after commit, read in callbacks).
  const latest = useRef({ collectionName, previewData, pins, options });
  useLayoutEffect(() => {
    latest.current = { collectionName, previewData, pins, options };
  });

  const setTopicCount = useCallback((nTopics: number) => {
    if (!useVisualizationStore.getState().topicCut.previewing) latest.current.options.onPreviewStart?.();
    setTopicCut({ nTopics, previewing: true });
  }, [setTopicCut]);
  const reset = useCallback(() => setTopicCut({ previewing: false, nTopics: null }), [setTopicCut]);
  const setLinkage = useCallback((linkage: TopicTreeLinkage) => setTopicCut({ linkage }), [setTopicCut]);
  const setAbsorbNoise = useCallback((absorb: boolean) => setTopicCut({ absorbNoise: absorb }), [setTopicCut]);

  const [reduceMutation] = useMutation<{ reduceTopics: ReduceTopicsResult }>(REDUCE_TOPICS);
  const [applying, setApplying] = useState(false);
  // Keyed by collection: another collection never shows this one's error.
  const [applyFailure, setApplyFailure] = useState<{ collection: string; message: string } | null>(null);
  const applyError = applyFailure?.collection === collectionName ? applyFailure.message : null;

  const apply = useCallback(async (regenerateLabels: boolean) => {
    if (!collectionName || !view || !previewing) return;
    const collection = collectionName;
    const input = ReductionInputs.reduceTopics(
      collection,
      {
        strategy: cut.linkage,
        method: 'fixed_n',
        nTopics: topicCount,
        distanceThreshold: DEFAULT_DISTANCE_THRESHOLD,
        absorbNoise,
      },
      { regenerateLabels },
      view.availableLinkages.includes('density'),
    );
    const fail = (message: string) => setApplyFailure({ collection, message });
    setApplying(true);
    setApplyFailure(null);
    try {
      const { data: response, error } = await reduceMutation({ variables: { input } });
      if (error) throw error;
      const result = response?.reduceTopics ?? null;
      const outcome = announceJobOutcome(result, {
        cancelled: () => `Applying the topic cut for "${collection}" was cancelled. The saved topics are unchanged.`,
        failed: (r) => `Applying the topic cut failed: ${r.error}`,
        succeeded: (r) => `Saved ${r.numTopicsAfter} topics for "${r.collectionName}"`,
      }, toast);
      if (outcome !== 'succeeded') {
        fail(result?.error ?? 'The reduction returned no result');
        return;
      }
      if (latest.current.collectionName !== collection) return;

      // Apply renumbered the topics and made the base topics subtopics: reload
      // everything derived from them before leaving the (frozen) preview.
      setFrozen({ collection, data: latest.current.previewData, pins: latest.current.pins });
      const reloads = await Promise.allSettled([
        latest.current.options.onApplied?.(),
        refetch(),
        refreshTopicSimilarityOrders(client),
      ]);
      const failed = reloads.find((r): r is PromiseRejectedResult => r.status === 'rejected');
      if (failed) {
        const reason = failed.reason instanceof Error ? failed.reason.message : String(failed.reason);
        fail(`Saved, but reloading the topics failed (${reason}). Reload the page.`);
      }
      if (latest.current.collectionName === collection) setTopicCut({ previewing: false, nTopics: null });
      setFrozen(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to apply the topic cut';
      fail(message);
      toast.error(message);
    } finally {
      setApplying(false);
    }
  }, [collectionName, view, previewing, cut.linkage, topicCount, absorbNoise, reduceMutation, refetch, client, setTopicCut]);

  return {
    view,
    loading,
    error: queryError?.message ?? parsed.error,
    linkage: cut.linkage,
    setLinkage,
    absorbNoise: cut.absorbNoise,
    setAbsorbNoise,
    minTopics: MIN_TOPICS,
    maxTopics,
    topicCount,
    savedTopicCount,
    setTopicCount,
    previewing: previewing || frozen !== null,
    liveSlider: (data?.ids.length ?? 0) <= TopicCutPreview.LIVE_SLIDER_MAX_ITEMS,
    reset,
    previewData,
    apply,
    applying,
    applyError,
    jobId: collectionName ? JobIds.topicReduction(collectionName) : null,
  };
}
