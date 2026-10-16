'use client';

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useApolloClient, useMutation, useQuery } from '@apollo/client/react';

import {
  GET_COLLECTION_PROBES,
  GET_PROBE_CLASS_PROBABILITIES,
  GET_PROBE_SCORES,
} from '../graphql/queries';
import { DELETE_PROBE, TRAIN_PROBE, type TrainProbeResult } from '../graphql/mutations';
import { useVisualizationStore } from '../stores/useVisualizationStore';
import type { EmbeddingData } from '../types/types';
import type { ColorFieldOption } from '../utils/fieldAnalysis';
import {
  ProbeFieldNames,
  buildProbeColorAliases,
  buildProbeFieldOptions,
  mergeProbeScores,
  parseProbeClassField,
  pendingRecolorAction,
  type PendingRecolor,
  type ProbeClassProbabilitiesData,
  type ProbeInfo,
  type ProbeScoresData,
  type ProbeWithScores,
} from '../utils/probeFields';
import type { TrainProbeInputVars } from '../utils/probeParams';

interface CollectionProbesData {
  collectionProbes: {
    collectionName: string;
    probes: ProbeInfo[];
  };
}

interface ProbeScoresQueryData {
  probeScores: ProbeScoresData | null;
}

interface ProbeClassProbabilitiesQueryData {
  probeClassProbabilities: ProbeClassProbabilitiesData | null;
}

const probeKey = (targetField: string, kind: string) => `${targetField}::${kind}`;

/** Drop one probe's entry from a per-probe state map (retrain / delete). */
function withoutKey<T>(prev: Record<string, T>, key: string): Record<string, T> {
  if (!(key in prev)) return prev;
  const next = { ...prev };
  delete next[key];
  return next;
}

export interface UseProbesReturn {
  /** The collection these probes belong to (null when none loaded). */
  collectionName: string | null;
  /** Trained probes for the loaded collection. */
  probes: ProbeInfo[];
  /**
   * Collection data with probe score/residual fields merged into a fresh
   * itemMetadata array, or null when there is nothing to merge. Feed this
   * (?? data) ONLY into useVisualizationPoints — metadata filters must keep
   * seeing the server-side fields.
   */
  augmentedData: EmbeddingData | null;
  /** ColorFieldOption entries for probe-derived fields. */
  fieldOptions: ColorFieldOption[];
  /** The collection's server-side item metadata (profiles probe targets). */
  itemMetadata: Record<string, unknown>[] | null;
  train: (input: TrainProbeInputVars) => Promise<void>;
  deleteProbe: (probe: ProbeInfo) => Promise<void>;
  /**
   * Color the map by P(class) of a multiclass probe, fetching that class's
   * probabilities first if they aren't loaded (one class per request).
   */
  showClassProbability: (probe: ProbeInfo, classIndex: number) => void;
  training: boolean;
  trainingError: string | null;
  /** Progress-subscription job id, set while a training run is in flight. */
  jobId: string | null;
}

/**
 * Owns probe state for the visualization dashboard: the trained-probes list,
 * per-probe score arrays (merged client-side into item metadata — the
 * persisted field_analysis cache is deliberately bypassed), training and
 * deletion. Auto-recolors to the new score field (multiclass: predicted
 * class) once its scores are loaded. Multiclass class probabilities are
 * fetched lazily, one class at a time, when a P(class) field is requested —
 * from the probe row or from a colorBy URL param.
 */
export function useProbes(
  collectionName: string | null,
  data: EmbeddingData | null,
): UseProbesReturn {
  const [scoresByKey, setScoresByKey] = useState<Record<string, ProbeScoresData>>({});
  // Lazily loaded class probabilities: probe key -> class index -> values.
  const [classProbsByKey, setClassProbsByKey] = useState<
    Record<string, Record<number, ProbeClassProbabilitiesData>>
  >({});
  const classFetchesRef = useRef(new Set<string>());
  // Per-probe generation, bumped on retrain/delete: a class-probability fetch
  // that started before must not write the old model's values back.
  const generationRef = useRef<Record<string, number>>({});
  const [training, setTraining] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  // Progress-subscription job id for the in-flight run, pinned to the
  // collection the run started for (not the live one).
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  // Field (and scale) to auto-recolor to once its values have landed.
  const pendingRecolorRef = useRef<PendingRecolor | null>(null);
  // Live collection, so a train that settles after a collection switch can
  // tell it no longer owns the hook state (field names are collection-
  // agnostic, so acting on a stale result would recolor/clobber the new
  // collection's probes).
  const collectionRef = useRef(collectionName);
  collectionRef.current = collectionName;

  const { data: probesData, refetch: refetchProbes } = useQuery<CollectionProbesData>(
    GET_COLLECTION_PROBES,
    {
      variables: { collectionName },
      skip: !collectionName,
      fetchPolicy: 'no-cache',
    },
  );
  const probes = useMemo(
    () => probesData?.collectionProbes.probes ?? [],
    [probesData],
  );

  // Direct client.query, NOT useLazyQuery: a single lazy-query executor
  // aborts its previous in-flight call when re-invoked, so fanning the
  // initial load's N score fetches through one executor cancelled all but
  // the last — and the batch-level catch then dropped even that. Independent
  // client queries run concurrently without shared state.
  // no-cache: retrained probes reuse identical query args, and score arrays
  // are too large to duplicate in Apollo's normalized cache.
  const client = useApolloClient();
  const fetchScores = useCallback(
    (variables: { collectionName: string; targetField: string; kind: string }) =>
      client.query<ProbeScoresQueryData>({
        query: GET_PROBE_SCORES,
        variables,
        fetchPolicy: 'no-cache',
      }),
    [client],
  );
  const [trainProbeMutation] = useMutation<{ trainProbe: TrainProbeResult }>(TRAIN_PROBE);
  const [deleteProbeMutation] = useMutation<{ deleteProbe: boolean }>(DELETE_PROBE);

  // Reset per-collection state when the collection changes.
  useEffect(() => {
    setScoresByKey({});
    setClassProbsByKey({});
    classFetchesRef.current.clear();
    generationRef.current = {};
    setTrainingError(null);
    pendingRecolorRef.current = null;
  }, [collectionName]);

  // Fetch score arrays for probes that don't have them yet (initial load and
  // after refetches). Batched into a single state update.
  useEffect(() => {
    if (!collectionName || probes.length === 0) return;
    const missing = probes.filter((p) => !scoresByKey[probeKey(p.targetField, p.kind)]);
    if (missing.length === 0) return;

    let cancelled = false;
    Promise.all(
      missing.map(async (p) => {
        // Per-probe isolation: one failed fetch must not drop the batch —
        // that probe just stays listed without coloring.
        try {
          const result = await fetchScores({
            collectionName,
            targetField: p.targetField,
            kind: p.kind,
          });
          return [probeKey(p.targetField, p.kind), result.data?.probeScores ?? null] as const;
        } catch {
          return [probeKey(p.targetField, p.kind), null] as const;
        }
      }),
    ).then((entries) => {
      if (cancelled) return;
      const loaded = entries.filter(([, scores]) => scores !== null);
      if (loaded.length === 0) return;
      setScoresByKey((prev) => {
        const next = { ...prev };
        for (const [key, scores] of loaded) {
          next[key] = scores as ProbeScoresData;
        }
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [collectionName, probes, scoresByKey, fetchScores]);

  const probesWithScores: ProbeWithScores[] = useMemo(
    () =>
      probes.flatMap((probe) => {
        const key = probeKey(probe.targetField, probe.kind);
        const scores = scoresByKey[key];
        return scores ? [{ probe, scores, classProbabilities: classProbsByKey[key] }] : [];
      }),
    [probes, scoresByKey, classProbsByKey],
  );

  const augmentedData = useMemo(() => {
    if (!data || probesWithScores.length === 0) return null;
    return {
      ...data,
      itemMetadata: mergeProbeScores(data.itemMetadata, data.ids, probesWithScores),
    };
  }, [data, probesWithScores]);

  const fieldOptions = useMemo(
    () => buildProbeFieldOptions(probesWithScores, data?.itemMetadata),
    [probesWithScores, data],
  );

  // Predicted-class fields borrow their target field's colors. Published
  // before paint so the first frame colored by a predicted field is right.
  const itemMetadata = data?.itemMetadata;
  const colorAliases = useMemo(
    () => (itemMetadata ? buildProbeColorAliases(probes, itemMetadata) : {}),
    [probes, itemMetadata],
  );
  useLayoutEffect(() => {
    useVisualizationStore.getState().setCategoryColorAliases(colorAliases);
  }, [colorAliases]);

  // Auto-recolor once the pending field's values are merged — unless the
  // user picked another color field while they were loading.
  const colorByField = useVisualizationStore((s) => s.colorByField);
  useEffect(() => {
    const pending = pendingRecolorRef.current;
    if (!pending) return;
    const action = pendingRecolorAction(pending, colorByField, fieldOptions);
    if (action === 'wait') return;
    pendingRecolorRef.current = null;
    if (action === 'apply') {
      useVisualizationStore.getState().setColorByField(pending.field, pending.scale);
    }
  }, [fieldOptions, colorByField]);

  /** Queue a recolor to `field` for when its values land. */
  const queueRecolor = useCallback((field: string, scale: PendingRecolor['scale']) => {
    pendingRecolorRef.current = {
      field,
      scale,
      fromField: useVisualizationStore.getState().colorByField,
    };
  }, []);

  const ensureClassProbabilities = useCallback(
    async (probe: ProbeInfo, classIndex: number) => {
      if (!collectionName) return;
      const key = probeKey(probe.targetField, probe.kind);
      const generation = generationRef.current[key] ?? 0;
      const fetchKey = `${key}::${generation}::${classIndex}`;
      if (classProbsByKey[key]?.[classIndex] || classFetchesRef.current.has(fetchKey)) return;
      classFetchesRef.current.add(fetchKey);
      const startedFor = collectionName;
      try {
        const result = await client.query<ProbeClassProbabilitiesQueryData>({
          query: GET_PROBE_CLASS_PROBABILITIES,
          variables: {
            collectionName,
            targetField: probe.targetField,
            kind: probe.kind,
            classIndex,
          },
          fetchPolicy: 'no-cache',
        });
        const values = result.data?.probeClassProbabilities;
        const stale =
          collectionRef.current !== startedFor || (generationRef.current[key] ?? 0) !== generation;
        if (!values || stale) return;
        setClassProbsByKey((prev) => ({
          ...prev,
          [key]: { ...prev[key], [classIndex]: values },
        }));
      } catch {
        // The probe stays usable; picking the class again retries.
      } finally {
        classFetchesRef.current.delete(fetchKey);
      }
    },
    [client, collectionName, classProbsByKey],
  );

  const showClassProbability = useCallback(
    (probe: ProbeInfo, classIndex: number) => {
      const field = new ProbeFieldNames(probe).classProbability(classIndex);
      if (classProbsByKey[probeKey(probe.targetField, probe.kind)]?.[classIndex]) {
        useVisualizationStore.getState().setColorByField(field, 'sequential');
        return;
      }
      queueRecolor(field, 'sequential');
      void ensureClassProbabilities(probe, classIndex);
    },
    [classProbsByKey, ensureClassProbabilities, queueRecolor],
  );

  // A P(class) color field can arrive without a click (shared URL, preset):
  // fetch its class probabilities so it doesn't render as all-missing.
  useEffect(() => {
    if (!colorByField) return;
    const parsed = parseProbeClassField(colorByField, probes);
    if (parsed) void ensureClassProbabilities(parsed.probe, parsed.classIndex);
  }, [colorByField, probes, ensureClassProbabilities]);

  const train = useCallback(
    async (input: TrainProbeInputVars) => {
      if (!collectionName) return;
      const { targetField, kind } = input;
      const startedFor = collectionName;
      const stillCurrent = () => collectionRef.current === startedFor;
      setTraining(true);
      setTrainingError(null);
      setActiveJobId(`${startedFor}_probe`);
      try {
        const { data: res, error: mutationError } = await trainProbeMutation({
          variables: { input },
          // Mirrors the long-running-mutation convention (extractTopics).
          context: { fetchOptions: { timeout: 600000 } },
        });
        if (!stillCurrent()) return;
        const result = res?.trainProbe;
        if (mutationError) {
          setTrainingError(mutationError.message);
        } else if (result?.error) {
          setTrainingError(result.error);
        } else if (result?.probe) {
          // Multiclass probes open on their predicted class; others on the score.
          const names = new ProbeFieldNames(result.probe);
          if (names.predicted) queueRecolor(names.predicted, 'categorical');
          else queueRecolor(names.score, 'sequential');
          // Force a fresh score fetch for this key (retrain replaces scores).
          const key = probeKey(targetField, kind);
          generationRef.current[key] = (generationRef.current[key] ?? 0) + 1;
          setScoresByKey((prev) => withoutKey(prev, key));
          setClassProbsByKey((prev) => withoutKey(prev, key));
        }
      } catch (err) {
        if (stillCurrent()) {
          setTrainingError(err instanceof Error ? err.message : 'Probe training failed');
        }
      } finally {
        // The run persists server-side even if the connection dropped, so
        // resync the probe list on settle (skipped after a collection switch —
        // the query already refetched for the new collection).
        if (stillCurrent()) {
          await refetchProbes().catch(() => {});
        }
        setTraining(false);
        setActiveJobId(null);
      }
    },
    [collectionName, trainProbeMutation, refetchProbes, queueRecolor],
  );

  const deleteProbe = useCallback(
    async (probe: ProbeInfo) => {
      if (!collectionName) return;
      const store = useVisualizationStore.getState();
      // Clear a dangling color field before its data disappears.
      if (store.colorByField && new ProbeFieldNames(probe).matches(store.colorByField)) {
        store.setColorByField(null);
      }
      try {
        await deleteProbeMutation({
          variables: {
            collectionName,
            targetField: probe.targetField,
            kind: probe.kind,
          },
        });
      } finally {
        const key = probeKey(probe.targetField, probe.kind);
        generationRef.current[key] = (generationRef.current[key] ?? 0) + 1;
        setScoresByKey((prev) => withoutKey(prev, key));
        setClassProbsByKey((prev) => withoutKey(prev, key));
        await refetchProbes().catch(() => {});
      }
    },
    [collectionName, deleteProbeMutation, refetchProbes],
  );

  return {
    collectionName,
    probes,
    augmentedData,
    fieldOptions,
    itemMetadata: data?.itemMetadata ?? null,
    train,
    deleteProbe,
    showClassProbability,
    training,
    trainingError,
    jobId: activeJobId,
  };
}
