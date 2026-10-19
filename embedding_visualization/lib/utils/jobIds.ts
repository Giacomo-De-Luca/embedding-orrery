/**
 * Job ids shared by progress subscriptions, the Jobs list and cancelJob.
 * Mirror of interpretability_backend/backend/services/job_ids.py — both sides
 * are tested against the same literal examples, so a modal always subscribes
 * to (and cancels) the id the backend reports under.
 */
export const JobIds = {
  embedding: (collectionName: string) => collectionName,
  // Same id as the embedding: the embed pipeline's topic step reports on it.
  topicExtraction: (collectionName: string) => collectionName,
  topicReduction: (collectionName: string) => `${collectionName}_reduce`,
  topicEvaluation: (collectionName: string) => `${collectionName}_evaluate`,
  llmLabeling: (collectionName: string) => `${collectionName}_llm_labeling`,
  saeActivations: (collectionName: string) => `${collectionName}_sae_activations`,
  probe: (collectionName: string) => `${collectionName}_probe`,
  saePrepare: (modelSize: string, variant: string, layer: number, hookType: string, width: string) =>
    `sae_prepare_${modelSize}_${variant}_${layer}_${hookType}_${width}`,
} as const;
