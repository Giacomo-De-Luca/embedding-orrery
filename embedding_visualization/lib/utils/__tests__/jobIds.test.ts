import { describe, it, expect } from 'vitest';
import { JobIds } from '../jobIds';

// Same literal examples as interpretability_backend/unit_tests/test_job_ids.py,
// so the two implementations cannot drift apart.
describe('JobIds', () => {
  it('builds collection-scoped ids like the backend', () => {
    expect(JobIds.embedding('col')).toBe('col');
    expect(JobIds.topicExtraction('col')).toBe('col');
    expect(JobIds.topicReduction('col')).toBe('col_reduce');
    expect(JobIds.topicEvaluation('col')).toBe('col_evaluate');
    expect(JobIds.llmLabeling('col')).toBe('col_llm_labeling');
    expect(JobIds.saeActivations('col')).toBe('col_sae_activations');
    expect(JobIds.probe('col')).toBe('col_probe');
  });

  it('builds the SAE prepare id like the backend', () => {
    expect(JobIds.saePrepare('4b', 'it', 9, 'resid_post', '16k')).toBe('sae_prepare_4b_it_9_resid_post_16k');
  });
});
