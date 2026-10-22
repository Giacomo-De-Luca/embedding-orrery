/**
 * Topic extraction inputs of the Collections page, as pure builders so node
 * tests can import them. `TopicConfigForm.tsx` re-exports `TopicConfigState`,
 * `DEFAULT_TOPIC_CONFIG` and `toTopicConfigInput`. The reduction rules and
 * request builders live in lib/utils/topicReduction.ts (shared with the
 * Explore topic slider) and are re-exported here.
 */
import type { TopicConfigInput } from '@/lib/graphql/mutations';
import { ReductionInputs, ReductionRules, DEFAULT_REDUCTION_CHOICE, type ReductionChoice } from '@/lib/utils/topicReduction';

export {
  DEFAULT_DISTANCE_THRESHOLD,
  DEFAULT_REDUCTION_CHOICE,
  REDUCTION_STRATEGY_OPTIONS,
  ReductionInputs,
  ReductionRules,
  type ReductionChoice,
  type ReductionFields,
  type ReductionLabelling,
} from '@/lib/utils/topicReduction';

// ========== Topic extraction form ==========

/** Stored visualization projections that can double as the clustering space. */
export const PROJECTION_SPACES = ['umap_3d', 'umap_2d', 'pca_3d', 'pca_2d'] as const;

/** Projection loaded for ids/documents (and used as fallback) when clustering on raw vectors. */
const RAW_MODE_PROJECTION = 'umap_3d';

export interface TopicConfigState {
  clusteringMethod: string;      // "hdbscan" | "kmeans" | "gmm" | "spectral"
  nClusters: number;
  minTopicSize: number;
  nKeywords: number;
  // Single space selector: a stored projection ("umap_3d" | "umap_2d" | "pca_3d" | "pca_2d")
  // or a raw-vector mode ("cluster_umap" | "embedding"). Split back into the GraphQL
  // projectionType + clusterOn pair in toTopicConfigInput.
  clusteringSpace: string;
  clusterNComponents: number;    // BERTopic UMAP dims (cluster_umap only)
  clusterMinDist: number;        // BERTopic UMAP min_dist (cluster_umap only)
  clusterNNeighbors: number;     // BERTopic UMAP n_neighbors (cluster_umap only)
  useLlmLabels: boolean;
  llmProvider: string;
  llmModel: string;
  // Reduction after extraction
  enableReduction: boolean;
  reduction: ReductionChoice;
}

export const DEFAULT_TOPIC_CONFIG: TopicConfigState = {
  clusteringMethod: 'hdbscan',
  nClusters: 10,
  minTopicSize: 10,
  nKeywords: 10,
  clusteringSpace: 'umap_3d',
  clusterNComponents: 5,
  clusterMinDist: 0.0,
  clusterNNeighbors: 15,
  useLlmLabels: false,
  llmProvider: 'gemini',
  llmModel: 'gemini-3-flash-preview',
  enableReduction: false,
  reduction: DEFAULT_REDUCTION_CHOICE,
};

/** Convert TopicConfigState to the GraphQL input shape. */
export function toTopicConfigInput(state: TopicConfigState): TopicConfigInput {
  const isProjectionSpace = (PROJECTION_SPACES as readonly string[]).includes(state.clusteringSpace);
  const config: TopicConfigInput = {
    minTopicSize: state.minTopicSize,
    nKeywords: state.nKeywords,
    projectionType: isProjectionSpace ? state.clusteringSpace : RAW_MODE_PROJECTION,
    useLlmLabels: state.useLlmLabels,
    clusteringMethod: state.clusteringMethod,
    clusterOn: isProjectionSpace ? 'projection' : state.clusteringSpace,
  };
  if (state.clusteringMethod !== 'hdbscan') {
    config.nClusters = state.nClusters;
  }
  if (state.clusteringSpace === 'cluster_umap') {
    config.clusterNComponents = state.clusterNComponents;
    config.clusterMinDist = state.clusterMinDist;
    config.clusterNNeighbors = state.clusterNNeighbors;
  }
  if (state.useLlmLabels) {
    config.llmProvider = state.llmProvider;
    config.llmModel = state.llmModel;
  }
  if (state.enableReduction) {
    config.reduction = { enabled: true, ...ReductionInputs.fields(state.reduction, ReductionRules.densityAvailableFor(state.clusteringMethod)) };
  }
  return config;
}
