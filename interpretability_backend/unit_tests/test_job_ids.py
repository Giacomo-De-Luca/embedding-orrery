"""JobIds: the single source of progress / job-state / cancel ids.

The frontend mirror (embedding_visualization/lib/utils/jobIds.ts) is tested
against the same literal examples in lib/utils/__tests__/jobIds.test.ts.
"""

from backend.services.job_ids import JobIds


def test_collection_scoped_ids():
    assert JobIds.embedding("col") == "col"
    assert JobIds.topic_extraction("col") == "col"
    assert JobIds.topic_reduction("col") == "col_reduce"
    assert JobIds.topic_evaluation("col") == "col_evaluate"
    assert JobIds.llm_labeling("col") == "col_llm_labeling"
    assert JobIds.sae_activations("col") == "col_sae_activations"
    assert JobIds.probe("col") == "col_probe"


def test_sae_prepare_id():
    assert (
        JobIds.sae_prepare("4b", "it", 9, "resid_post", "16k")
        == "sae_prepare_4b_it_9_resid_post_16k"
    )
