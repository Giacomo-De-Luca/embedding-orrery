# Topic Semantic Search and Topic-Scoped Search

The Topics section of the Explore page's Search panel has two modes:

- **Browse**: filter topics by label/keyword substring (client-side).
- **Semantic**: "describe a topic" — rank the collection's topics by similarity
  to a text query.

Selecting topics in either mode also **scopes** the header's semantic search
(and text search) to the selected topics.

## Why topic centroids

Topic membership lives only in DuckDB `topic_assignments`: topic fields are
stripped from item metadata on insert, and ChromaDB stores vectors only. The
original implementation grouped semantic-search hits by `metadata.topic_id`
(never present, so the ranking was always empty) and filtered on
`topic_id` in item JSON (never present, so topic-scoped search returned
nothing). Semantic topic *reduction* had the same flaw: it looked members up
with ChromaDB `where={"topic_id": …}`, got nothing, and averaged zero vectors.
Reduction now cuts merge trees built from these centroids
([TOPIC_MERGE_TREES.md](TOPIC_MERGE_TREES.md)).

Topic search now compares the query with each topic's **embedding centroid**
(the mean of its members' vectors), computed once per extraction and stored in
DuckDB:

- same vector space as the query (the collection's own model, query mode);
- no extra embedding calls (labels would need one per topic, metered for
  Gemini/OpenAI collections, impossible in the read-only demo);
- independent of label quality (c-TF-IDF keyword labels embed poorly);
- every topic gets a score, not only those that happen to appear among the
  top document hits.

The scatter plots keep computing their own **projection-space** centroids
(`clusterGeometry.ts`, filter-aware, for haze and cluster labels); embedding
centroids never leave the backend — only similarity scores do.

## Storage

`topic_centroids` (DuckDB, FK to `topic_extractions`):

| Column | Meaning |
|---|---|
| `extraction_id`, `level`, `topic_id` | PK; `level` is `topic` or `subtopic` |
| `parent_topic_id` | subtopic rows: the topic the subtopic belongs to |
| `member_count` | items averaged |
| `centroid` | `FLOAT[]`, mean of the L2-normalised member vectors (a mean direction; its length is ≤ 1) |

Subtopic rows exist after a reduction (subtopics = the pre-reduction clusters
recorded in `topic_assignments.subtopic_id`). A new table rather than a
`topic_info` column keeps the documented WAL-replay `ALTER` crash out of play.
Seed snapshots copy it with the other topic child tables (skipped, count 0,
when the source DB predates the table); `delete_dataset` removes it.
`replace_topic_centroids` swaps an extraction's rows in one transaction.

A new extraction becomes active before its centroids are written, so a topic
search during that short window reports "not computed".

## Computing

`TopicCentroidService` (`backend/services/topic_centroid_service.py`) is the
single code path, built on the numpy-only accumulator in
`backend/utils/topic_centroids.py`:

- Member vectors are **L2-normalised before averaging** and the query is
  normalised before ranking: some embedders do not normalise their output
  (QWEN with `normalize=False`, which also prefixes queries only), and with
  mean-centring a length mismatch between query and documents would add a
  component along the origin and reorder topics.
- `refresh(collection, extraction_id, preloaded=None)` reads the extraction's
  assignments, streams vectors (`embedding_loader.iter_embedding_batches`,
  memory = topics × dim) or walks `preloaded` `(ids, vectors)` in 5k-row
  slices (bounding per-step copies), and replaces the stored rows. Called after `extract_topics` (reusing the raw vectors already
  loaded for the default `cluster_umap`/`embedding` clustering) and after
  `reduce_existing_topics`. A failure is logged and never fails the job.
- `tables_for_assignments(collection, [item_topics, …], preloaded)` returns
  one in-memory `CentroidTable` per assignment map from a single pass over the
  vectors (`refresh` uses it too). Topic reduction builds its merge trees from
  these tables, or from the stored ones, in the centred space below
  (`TopicMergeTreeService`, see [TOPIC_MERGE_TREES.md](TOPIC_MERGE_TREES.md)).
- `centroids_for_assignments(collection, item_topics, order)` returns the raw
  centroid matrix the legacy semantic reduction (`legacy_semantic`) receives;
  if it cannot be computed, that reduction falls back to c-TF-IDF with a
  logged warning and a progress message.
- `stored_table(extraction_id, level)` reads one stored level by extraction id.

**Backfill** for extractions created before centroids existed (backend stopped):

```bash
uv run python -m interpretability_backend.scripts.backfill_topic_centroids
```

Config: `interpretability_backend/scripts/backfill_topic_centroids_config.toml`
(`collections = "all"` or a list; `skip_existing`). Rebuild the seed snapshots
afterwards so the demo gets centroids too.

## Ranking

`TopicCentroidRanker.rank(query, topics, subtopics, limit)`:

1. **Centring** (default on): the origin is the count-weighted mean of all
   topic centroids, noise included (= the dataset mean). Averages of many
   vectors drift toward it, so raw cosines bunch together; the query and the
   centroids are both shifted by it before the cosine.
2. **Subtopic roll-up**: a topic scores the max of its own centroid and its
   subtopics'; when a subtopic wins, the result names it
   (`matchedSubtopicId/Label`, shown as "via …" in the UI). This keeps a
   merged topic's blurry centroid from hiding a strand that matches well.
3. Noise (`-1`) is never returned; ties break by topic id.

The centring is reusable for anything else that compares topic centroids
(e.g. ordering topics by similarity), so it cannot drift from search:
`CENTER_BY_DEFAULT` is the single on/off default, `ranker.origin(topics)` gives
the origin (use the topic table's origin for subtopics too),
`CentroidTable.unit_directions(origin)` gives the centred unit rows whose dot
products are the cosines `rank()` uses, and
`TopicCentroidService.active_centroids(collection)` returns the active
extraction's `(topics, subtopics)` tables; `TopicCentroidService.ranker` is
the instance search ranks with.

GraphQL: `topicSemanticSearch(collectionName, query, limit = 20, queryPrompt)`
→ `[TopicSimilarityResult {topicId, similarity, matchedSubtopicId,
matchedSubtopicLabel}]`. The resolver is async (`asyncio.to_thread`); the query
is embedded by `ChromaDBClient.embed_query`, which raises instead of silently
falling back to ChromaDB's default embedder. Missing centroids or a dimension
mismatch return a GraphQL error the Topics section displays.

## Topic-scoped search

Filters on `topic_id`, `topic_label`, `subtopic_id`, `subtopic_label` resolve
through `topic_assignments` of the collection's active extraction
(`DuckDBClient._build_metadata_where`); all other fields keep the
item-metadata JSON path. This applies to `semanticSearch`,
`semanticSearchById`, `textSearch` and `embeddings`.

Semantic search no longer over-fetches and post-filters: the allowed IDs
(`get_filtered_item_ids`, no row cap) are passed to ChromaDB
`query(ids=…)`, so a filtered search returns the full `nResults` whenever that
many items match. This also fixed the date-filter undercount (a `year` filter
used to return only the matches among the 5× unscoped neighbours).

chromadb raises `InternalError` when the scope contains an ID it holds no
vector for (dataset items skipped at embed time, partial or resumed jobs).
`ChromaDBClient.semantic_search` then prunes the scope with chunked
`get(ids=…)` lookups (10k per call; one call over ~100k IDs exceeds SQLite's
variable limit) and retries once, so the common path stays a single query.

Range operators compare topic ids numerically and labels as text.

## Frontend

- `lib/hooks/useTopicSearch.ts` calls `TOPIC_SEMANTIC_SEARCH` (`no-cache`,
  stale responses dropped by `LatestRequestGate`), exposes
  `semanticResults` (`null` until a search completes) and `semanticError`, and
  **clears selection and results when the collection changes** — a carried-over
  selection would otherwise scope the next collection's searches to unrelated
  topic ids.
- `lib/utils/topicSearchResults.ts`: `toTopicSearchResults` (maps rows to
  `TopicInfo`, display relevance relative to the best score) and
  `buildTopicFilters`.
- `lib/utils/queryPrompt.ts`: `resolveQueryPrompt`, shared with `useAppSearch`.
- `app/components/TopicSearchSection.tsx`: relevance bars, "via <subtopic>",
  "No matching topics" and error states.
