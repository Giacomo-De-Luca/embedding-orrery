# Topic extraction

Clustering, keywords, labels and topic reduction: the computation behind
topic extraction. Orchestration and persistence live in
`backend/services/topic_extraction_service.py` (extraction, standalone
reduction), `topic_reduction_service.py` (reduction plans) and
`topic_merge_tree_service.py` (merge trees for the Explore slider).

Everything here imports hdbscan / sklearn / scipy, so modules in the GraphQL
schema's import graph import it lazily (`__init__.py` re-exports
`GenerateTopics` and `ClassTfidfTransformer` through a PEP 562 `__getattr__`;
`unit_tests/test_torch_free_import.py` guards it).

## Structure

| File | Purpose |
|---|---|
| `cluster_and_label.py` | `cluster_points(points, method, ...)` → `ClusteringOutcome(labels, density)`: HDBSCAN / k-means / GMM / spectral labels, plus HDBSCAN's density merge tree (`utils/topic_merge_tree.DensityTreeBuilder`), built in the process that fitted the model. `CtfidfKeywords`: the single c-TF-IDF keyword implementation (topics' documents joined into one class document, `CountVectorizer` + `ClassTfidfTransformer`) → `CtfidfResult(topics_data, topic_ids, matrix, words)`. `GenerateTopics`: clustering + keywords for one extraction. |
| `topic_reducer.py` | `TopicReducer`: `reduce_with_tree(tree, n_topics \| distance_threshold, noise_merges, absorb_noise)` cuts a merge tree over the topics (groups numbered by size, keywords re-extracted, `topic_hierarchy` new id → base ids). Legacy, kept only for comparison: `reduce_to_n_topics` (agglomerative, average linkage) and `auto_reduce_topics` (HDBSCAN on the topic vectors), on c-TF-IDF rows or raw centroids. `TopicReductionResult` is the shared result. |
| `llm_labeling.py` | Provider-agnostic LLM labels (Gemini / OpenAI, tenacity retries, cancel between topics): `generate_llm_labels(topics_data, documents_df, ...)` labels every non-noise topic in `topics_data`. |
| `extract_topics_bertopics.txt` | Inert BERTopic reference snippet (not importable). |

## Data structures

- `documents_df`: one row per document: `Document_ID` (position), `Document`
  (text), `Topic` (cluster id, `-1` = noise).
- `topics_data`: `{topic_id: [(word, score), ...]}`, top c-TF-IDF keywords.
- Merge trees (`utils/topic_merge_tree.py`): leaves are base topics, merges in
  scipy's linkage convention; see `documentation/TOPIC_MERGE_TREES.md`.
