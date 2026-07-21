'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLazyQuery, useMutation, useQuery } from '@apollo/client/react';

import { GET_COLLECTION_PROBES, GET_PROBE_SCORES } from '../graphql/queries';
import { DELETE_PROBE, TRAIN_PROBE, type TrainProbeResult } from '../graphql/mutations';
import { useVisualizationStore } from '../stores/useVisualizationStore';
import type { EmbeddingData } from '../types/types';
import type { ColorFieldOption } from '../utils/fieldAnalysis';
import {
  buildProbeFieldOptions,
  mergeProbeScores,
  probeAbsErrorField,
  probeConfusionField,
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

const probeKey = (targetField: string, kind: string) => `${targetField}::${kind}`;

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
  /** ColorFieldOption entries for probe score/residual fields. */
  fieldOptions: ColorFieldOption[];
  train: (input: TrainProbeInputVars) => Promise<void>;
  deleteProbe: (probe: ProbeInfo) => Promise<void>;
  training: boolean;
  trainingError: string | null;
  /** Progress-subscription job id, set while a training run is in flight. */
  jobId: string | null;
}

/**
 * Owns probe state for the visualization dashboard: the trained-probes list,
 * per-probe score arrays (merged client-side into item metadata — the
 * persisted field_analysis cache is deliberately bypassed), training and
 * deletion. Auto-recolors to the new score field once its scores are loaded.
 */
export function useProbes(
  collectionName: string | null,
  data: EmbeddingData | null,
): UseProbesReturn {
  const [scoresByKey, setScoresByKey] = useState<Record<string, ProbeScoresData>>({});
  const [training, setTraining] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  // Progress-subscription job id for the in-flight run, pinned to the
  // collection the run started for (not the live one).
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  // Score field to auto-recolor to once its scores have landed.
  const pendingRecolorRef = useRef<string | null>(null);
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

  // no-cache: retrained probes reuse identical query args, and score arrays
  // are too large to duplicate in Apollo's normalized cache.
  const [fetchScores] = useLazyQuery<ProbeScoresQueryData>(GET_PROBE_SCORES, {
    fetchPolicy: 'no-cache',
  });
  const [trainProbeMutation] = useMutation<{ trainProbe: TrainProbeResult }>(TRAIN_PROBE);
  const [deleteProbeMutation] = useMutation<{ deleteProbe: boolean }>(DELETE_PROBE);

  // Reset per-collection state when the collection changes.
  useEffect(() => {
    setScoresByKey({});
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
        const result = await fetchScores({
          variables: {
            collectionName,
            targetField: p.targetField,
            kind: p.kind,
          },
        });
        return [probeKey(p.targetField, p.kind), result.data?.probeScores ?? null] as const;
      }),
    )
      .then((entries) => {
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
      })
      .catch(() => {
        /* per-probe score fetch failures leave the probe listed without coloring */
      });
    return () => {
      cancelled = true;
    };
  }, [collectionName, probes, scoresByKey, fetchScores]);

  const probesWithScores: ProbeWithScores[] = useMemo(
    () =>
      probes.flatMap((probe) => {
        const scores = scoresByKey[probeKey(probe.targetField, probe.kind)];
        return scores ? [{ probe, scores }] : [];
      }),
    [probes, scoresByKey],
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

  // Auto-recolor once the freshly trained probe's scores are merged.
  useEffect(() => {
    const field = pendingRecolorRef.current;
    if (!field) return;
    if (!fieldOptions.some((o) => o.field === field)) return;
    pendingRecolorRef.current = null;
    useVisualizationStore.getState().setColorByField(field, 'sequential');
  }, [fieldOptions]);

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
          pendingRecolorRef.current = result.probe.scoreField;
          // Force a fresh score fetch for this key (retrain replaces scores).
          setScoresByKey((prev) => {
            const next = { ...prev };
            delete next[probeKey(targetField, kind)];
            return next;
          });
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
    [collectionName, trainProbeMutation, refetchProbes],
  );

  const deleteProbe = useCallback(
    async (probe: ProbeInfo) => {
      if (!collectionName) return;
      const store = useVisualizationStore.getState();
      // Clear a dangling color field before its data disappears.
      const probeFields = [
        probe.scoreField,
        probe.residualField,
        probeAbsErrorField(probe),
        probeConfusionField(probe),
      ];
      if (store.colorByField && probeFields.includes(store.colorByField)) {
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
        setScoresByKey((prev) => {
          const next = { ...prev };
          delete next[probeKey(probe.targetField, probe.kind)];
          return next;
        });
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
    train,
    deleteProbe,
    training,
    trainingError,
    jobId: activeJobId,
  };
}
