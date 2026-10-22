import { describe, expect, it } from 'vitest';
import {
  DEFAULT_REDUCTION_CHOICE,
  DEFAULT_TOPIC_CONFIG,
  REDUCTION_STRATEGY_OPTIONS,
  ReductionInputs,
  ReductionRules,
  toTopicConfigInput,
  type ReductionChoice,
  type ReductionFields,
} from '../topicConfigInput';
import type { ReductionStrategy } from '@/lib/graphql/mutations';

const STRATEGIES: ReductionStrategy[] = ['average', 'ward', 'density', 'legacy_ctfidf', 'legacy_semantic'];

/**
 * The backend's validation of a reduction request, mirrored from
 * interpretability_backend/backend/services/topic_reduction_types.py
 * (ReductionRequest.__post_init__ after ReductionRequest.create). Returns the
 * error the API would raise, or null when the request is accepted.
 */
function backendError(fields: ReductionFields): string | null {
  const strategy = fields.strategy;
  if (!STRATEGIES.includes(strategy)) return 'Unknown reduction strategy';
  if (fields.method !== 'fixed_n' && fields.method !== 'auto') return 'Unknown reduction method';
  if (fields.method === 'fixed_n') {
    if (!Number.isInteger(fields.nTopics) || (fields.nTopics as number) < 1) return 'A fixed_n reduction needs nTopics >= 1';
    if (fields.distanceThreshold != null) return 'distanceThreshold applies to the auto method only';
  } else if (!['average', 'legacy_ctfidf', 'legacy_semantic'].includes(strategy)) {
    return `The ${strategy} strategy needs a topic count (fixed_n)`;
  } else if (fields.distanceThreshold != null && strategy !== 'average') {
    return 'distanceThreshold applies to the average strategy only';
  }
  if (fields.distanceThreshold != null && !(fields.distanceThreshold >= 0 && fields.distanceThreshold <= 2)) {
    return 'distanceThreshold is a cosine distance: between 0 and 2';
  }
  if (fields.absorbNoise && strategy !== 'density') return 'absorbNoise needs the density strategy';
  return null;
}

const choice = (patch: Partial<ReductionChoice>): ReductionChoice => ({ ...DEFAULT_REDUCTION_CHOICE, ...patch });

describe('ReductionRules', () => {
  it('offers auto only for average and the legacy strategies', () => {
    expect(STRATEGIES.filter(ReductionRules.supportsAuto)).toEqual(['average', 'legacy_ctfidf', 'legacy_semantic']);
  });

  it('takes a distance threshold only for average and absorbs noise only for density', () => {
    expect(STRATEGIES.filter(ReductionRules.supportsThreshold)).toEqual(['average']);
    expect(STRATEGIES.filter(ReductionRules.supportsAbsorbNoise)).toEqual(['density']);
  });

  it('lists density only when the collection can use it', () => {
    expect(ReductionRules.strategies(true)).toEqual(STRATEGIES);
    expect(ReductionRules.strategies(false)).toEqual(STRATEGIES.filter((s) => s !== 'density'));
  });

  it('labels every strategy with colons, never dashes', () => {
    expect(REDUCTION_STRATEGY_OPTIONS.map((o) => o.value)).toEqual(STRATEGIES);
    expect(REDUCTION_STRATEGY_OPTIONS.find((o) => o.value === 'legacy_ctfidf')?.label).toBe('Legacy: c-TF-IDF');
    for (const option of REDUCTION_STRATEGY_OPTIONS) expect(option.label).not.toMatch(/[—–]/);
  });
});

describe('ReductionInputs.fields', () => {
  it('defaults to a fixed-N average cut', () => {
    expect(ReductionInputs.fields(DEFAULT_REDUCTION_CHOICE, true)).toEqual({
      strategy: 'average', method: 'fixed_n', nTopics: 10, absorbNoise: false,
    });
  });

  it('sends the distance threshold for an auto average cut, not a topic count', () => {
    expect(ReductionInputs.fields(choice({ method: 'auto', distanceThreshold: 0.55 }), true)).toEqual({
      strategy: 'average', method: 'auto', distanceThreshold: 0.55, absorbNoise: false,
    });
  });

  it('keeps the distance threshold a cosine distance (0 to 2)', () => {
    expect(ReductionInputs.fields(choice({ method: 'auto', distanceThreshold: 2.5 }), true).distanceThreshold).toBe(2);
    expect(ReductionInputs.fields(choice({ method: 'auto', distanceThreshold: -0.5 }), true).distanceThreshold).toBe(0);
  });

  it('runs legacy auto without a threshold', () => {
    expect(ReductionInputs.fields(choice({ strategy: 'legacy_ctfidf', method: 'auto' }), true)).toEqual({
      strategy: 'legacy_ctfidf', method: 'auto', absorbNoise: false,
    });
  });

  it('falls back to fixed N for strategies without auto', () => {
    expect(ReductionInputs.fields(choice({ strategy: 'ward', method: 'auto', nTopics: 7 }), true)).toEqual({
      strategy: 'ward', method: 'fixed_n', nTopics: 7, absorbNoise: false,
    });
  });

  it('absorbs noise only with density', () => {
    expect(ReductionInputs.fields(choice({ strategy: 'density', absorbNoise: true }), true).absorbNoise).toBe(true);
    expect(ReductionInputs.fields(choice({ strategy: 'ward', absorbNoise: true }), true).absorbNoise).toBe(false);
  });

  it('replaces density by average when the collection has no density tree', () => {
    expect(ReductionInputs.fields(choice({ strategy: 'density', absorbNoise: true }), false)).toEqual({
      strategy: 'average', method: 'fixed_n', nTopics: 10, absorbNoise: false,
    });
  });

  it('only ever builds requests the backend accepts', () => {
    for (const strategy of STRATEGIES) {
      for (const method of ['fixed_n', 'auto'] as const) {
        for (const absorbNoise of [false, true]) {
          for (const densityAvailable of [false, true]) {
            for (const distanceThreshold of [0.6, 2.5, -1]) {
              const fields = ReductionInputs.fields(
                choice({ strategy, method, absorbNoise, nTopics: 4, distanceThreshold }),
                densityAvailable,
              );
              const combination = { strategy, method, absorbNoise, densityAvailable, distanceThreshold };
              expect(backendError(fields), JSON.stringify(combination)).toBeNull();
            }
          }
        }
      }
    }
  });
});

describe('ReductionInputs.reduceTopics', () => {
  it('builds the standalone reduction input', () => {
    expect(ReductionInputs.reduceTopics('emotion', choice({ strategy: 'density', absorbNoise: true, nTopics: 6 }), {
      regenerateLabels: false,
    }, true)).toEqual({
      collectionName: 'emotion',
      strategy: 'density',
      method: 'fixed_n',
      nTopics: 6,
      absorbNoise: true,
      regenerateLabels: false,
    });
  });

  it('sends the LLM settings only when merged topics are relabelled', () => {
    const input = ReductionInputs.reduceTopics('emotion', DEFAULT_REDUCTION_CHOICE, {
      regenerateLabels: true, llmProvider: 'openai', llmModel: 'gpt-4o-mini',
    }, true);
    expect(input).toMatchObject({ regenerateLabels: true, llmProvider: 'openai', llmModel: 'gpt-4o-mini' });
    const plain = ReductionInputs.reduceTopics('emotion', DEFAULT_REDUCTION_CHOICE, {
      regenerateLabels: false, llmProvider: 'openai', llmModel: 'gpt-4o-mini',
    }, true);
    expect(plain).not.toHaveProperty('llmProvider');
    expect(plain).not.toHaveProperty('llmModel');
  });
});

describe('toTopicConfigInput', () => {
  it('splits a stored projection space into clusterOn + projectionType', () => {
    expect(toTopicConfigInput({ ...DEFAULT_TOPIC_CONFIG, clusteringSpace: 'pca_2d' })).toMatchObject({
      clusterOn: 'projection', projectionType: 'pca_2d',
    });
  });

  it('pins umap_3d for raw-vector modes and sends the BERTopic UMAP parameters', () => {
    const input = toTopicConfigInput({ ...DEFAULT_TOPIC_CONFIG, clusteringSpace: 'cluster_umap' });
    expect(input).toMatchObject({
      clusterOn: 'cluster_umap', projectionType: 'umap_3d',
      clusterNComponents: 5, clusterMinDist: 0, clusterNNeighbors: 15,
    });
  });

  it('sends nClusters only for the non-HDBSCAN methods and LLM settings only when labelling', () => {
    expect(toTopicConfigInput(DEFAULT_TOPIC_CONFIG)).not.toHaveProperty('nClusters');
    expect(toTopicConfigInput({ ...DEFAULT_TOPIC_CONFIG, clusteringMethod: 'kmeans', nClusters: 12 }).nClusters).toBe(12);
    expect(toTopicConfigInput(DEFAULT_TOPIC_CONFIG)).not.toHaveProperty('llmProvider');
  });

  it('leaves the reduction out unless enabled', () => {
    expect(toTopicConfigInput(DEFAULT_TOPIC_CONFIG)).not.toHaveProperty('reduction');
  });

  it('sends the inline reduction with the same fields as the standalone one', () => {
    const reduction = choice({ strategy: 'density', absorbNoise: true, nTopics: 8 });
    expect(toTopicConfigInput({ ...DEFAULT_TOPIC_CONFIG, enableReduction: true, reduction }).reduction).toEqual({
      enabled: true, strategy: 'density', method: 'fixed_n', nTopics: 8, absorbNoise: true,
    });
  });

  it('offers density inline only for HDBSCAN clustering', () => {
    const reduction = choice({ strategy: 'density', absorbNoise: true });
    const kmeans = toTopicConfigInput({
      ...DEFAULT_TOPIC_CONFIG, clusteringMethod: 'kmeans', enableReduction: true, reduction,
    });
    expect(kmeans.reduction).toMatchObject({ strategy: 'average', absorbNoise: false });
  });
});
