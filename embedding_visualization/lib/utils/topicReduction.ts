/**
 * Topic reduction strategies and the requests the backend accepts, shared by
 * the Collections page reduction forms (app/collections/lib/topicConfigInput.ts
 * re-exports them) and the Explore topic slider's Apply.
 *
 * The rules mirror the backend's validation in
 * interpretability_backend/backend/services/topic_reduction_types.py
 * (`ReductionRequest`): "auto" exists only for average and the legacy
 * strategies, a distance threshold only for auto + average, absorbNoise only
 * for density; topic counts never include noise.
 */
import type {
  ReduceTopicsInput,
  ReductionMethod,
  ReductionStrategy,
} from '../graphql/mutations';

/** Distance threshold of an "auto" average cut when none is given (backend config default). */
export const DEFAULT_DISTANCE_THRESHOLD = 0.7;
/** The threshold is a cosine distance; the backend rejects values outside this range. */
export const DISTANCE_THRESHOLD_RANGE = { min: 0, max: 2 } as const;

export const REDUCTION_STRATEGY_OPTIONS: ReadonlyArray<{
  value: ReductionStrategy;
  label: string;
  /** The Explore topic tree select's label. */
  shortLabel: string;
  hint: string;
}> = [
  {
    value: 'average',
    shortLabel: 'Average',
    label: 'Average (centred semantic)',
    hint: 'Merges topics whose centred embedding centroids are closest on average.',
  },
  {
    value: 'ward',
    shortLabel: 'Ward',
    label: 'Ward (size-weighted)',
    hint: 'Like average, but merging into a large topic costs more, so sizes stay balanced.',
  },
  {
    value: 'density',
    shortLabel: 'Density',
    label: 'Density (HDBSCAN tree)',
    hint: "Follows HDBSCAN's own cluster tree and can move noise into the topics it belongs to.",
  },
  {
    value: 'legacy_ctfidf',
    shortLabel: 'Legacy: c-TF-IDF',
    label: 'Legacy: c-TF-IDF',
    hint: 'The previous keyword-based merge, kept for comparison.',
  },
  {
    value: 'legacy_semantic',
    shortLabel: 'Legacy: semantic',
    label: 'Legacy: semantic',
    hint: 'The previous merge on raw embedding centroids, kept for comparison.',
  },
];

/** What a reduction form edits; `ReductionInputs` turns it into a valid request. */
export interface ReductionChoice {
  strategy: ReductionStrategy;
  method: ReductionMethod;
  /** Topics to keep, noise not counted (fixed_n). */
  nTopics: number;
  /** Merge while the centred cosine distance is at most this (auto + average). */
  distanceThreshold: number;
  /** Move noise into the topics that absorb it (density). */
  absorbNoise: boolean;
}

export const DEFAULT_REDUCTION_CHOICE: ReductionChoice = {
  strategy: 'average',
  method: 'fixed_n',
  nTopics: 10,
  distanceThreshold: DEFAULT_DISTANCE_THRESHOLD,
  absorbNoise: false,
};

/** Which options a strategy takes (the backend rejects the others). */
export class ReductionRules {
  static supportsAuto(strategy: ReductionStrategy): boolean {
    return strategy === 'average' || strategy === 'legacy_ctfidf' || strategy === 'legacy_semantic';
  }

  static supportsThreshold(strategy: ReductionStrategy): boolean {
    return strategy === 'average';
  }

  static supportsAbsorbNoise(strategy: ReductionStrategy): boolean {
    return strategy === 'density';
  }

  /** Only HDBSCAN builds a density tree, so an inline density reduction needs it. */
  static densityAvailableFor(clusteringMethod: string): boolean {
    return clusteringMethod === 'hdbscan';
  }

  /** Strategies a form offers; density needs an HDBSCAN density tree. */
  static strategies(densityAvailable: boolean): ReductionStrategy[] {
    return REDUCTION_STRATEGY_OPTIONS.map((o) => o.value).filter((s) => densityAvailable || s !== 'density');
  }

  /** The strategy a choice runs: density falls back to average when unavailable. */
  static effectiveStrategy(strategy: ReductionStrategy, densityAvailable: boolean): ReductionStrategy {
    return strategy === 'density' && !densityAvailable ? 'average' : strategy;
  }

  /** The method a choice runs: strategies without "auto" cut at a topic count. */
  static effectiveMethod(strategy: ReductionStrategy, method: ReductionMethod): ReductionMethod {
    return method === 'auto' && ReductionRules.supportsAuto(strategy) ? 'auto' : 'fixed_n';
  }
}

/** Reduction fields shared by `ReduceTopicsInput` and the inline `TopicReductionInput`. */
export type ReductionFields = Required<Pick<ReduceTopicsInput, 'strategy' | 'method' | 'absorbNoise'>> &
  Pick<ReduceTopicsInput, 'nTopics' | 'distanceThreshold'> & { strategy: ReductionStrategy };

/** How a standalone reduction labels merged topics. */
export interface ReductionLabelling {
  regenerateLabels: boolean;
  llmProvider?: string;
  llmModel?: string;
}

/** Builds reduction requests the backend accepts from a form's choice. */
export class ReductionInputs {
  static fields(choice: ReductionChoice, densityAvailable: boolean): ReductionFields {
    const strategy = ReductionRules.effectiveStrategy(choice.strategy, densityAvailable);
    const method = ReductionRules.effectiveMethod(strategy, choice.method);
    const fields: ReductionFields = {
      strategy,
      method,
      absorbNoise: ReductionRules.supportsAbsorbNoise(strategy) && choice.absorbNoise,
    };
    if (method === 'fixed_n') fields.nTopics = Math.max(1, Math.round(choice.nTopics));
    else if (ReductionRules.supportsThreshold(strategy)) {
      const { min, max } = DISTANCE_THRESHOLD_RANGE;
      fields.distanceThreshold = Math.min(Math.max(choice.distanceThreshold, min), max);
    }
    return fields;
  }

  /** `reduceTopics` input; LLM settings are sent only when merged topics are relabelled. */
  static reduceTopics(
    collectionName: string,
    choice: ReductionChoice,
    labelling: ReductionLabelling,
    densityAvailable: boolean,
  ): ReduceTopicsInput {
    const input: ReduceTopicsInput = {
      collectionName,
      ...ReductionInputs.fields(choice, densityAvailable),
      regenerateLabels: labelling.regenerateLabels,
    };
    if (labelling.regenerateLabels) {
      if (labelling.llmProvider) input.llmProvider = labelling.llmProvider;
      if (labelling.llmModel) input.llmModel = labelling.llmModel;
    }
    return input;
  }
}
