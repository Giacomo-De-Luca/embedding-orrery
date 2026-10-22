# Deleting a collection

The `deleteCollection` mutation removes a vector collection **and everything
derived from it**, through
`backend/services/collection_deletion.py::CollectionDeletionService`.

## What goes

| Owned by the collection | Where | Removed by |
|---|---|---|
| Vectors | ChromaDB collection | `ChromaDBClient.delete_collection` |
| HNSW index directory | `resources/vector_db/<vector segment id>/` | same (chromadb itself leaves it: `CHROMA_ORPHAN_INDEXES.md`) |
| Projections, topic extractions (info, assignments, centroids, merge trees), SAE document activations, probes and probe scores, registry row | DuckDB | `DuckDBClient.delete_vector_collection` |
| Dataset row, items table, full-text (BM25) index | DuckDB, only once no other collection uses the dataset | `DuckDBClient.delete_collection` → `delete_dataset` |
| Probe artifacts | `resources/probing_results/collections/<collection>/` (`probing_types.probe_collection_dir`, shared with `probing_service`) | the service |
| Job records (the resumable embedding and LLM-labeling jobs; the others write none) | `resources/job_state.json` | the service |
| Uploaded source file | `resources/uploads/<file>` | the service, only once its dataset is gone and nothing else uses it |

## Rules

- **Shared dataset.** A re-embedded collection shares its source's dataset:
  the items, the upload and the full-text index. Deleting one collection of
  the dataset leaves those for the others; they go with the last collection.
  Legacy datasets with no registered collection are deleted by name.
- **No job of the collection runs during a delete.**
  `cancel_registry.reserved_job_ids(JobIds.for_collection(name))` checks and
  reserves the collection's job ids under one lock, for the whole delete.
  - A job already running under one of them makes the delete fail with
    `CollectionBusyError`, a GraphQL error asking to cancel it first. The jobs
    that register: embeds, projections, topic extraction / reduction /
    evaluation, LLM labeling, SAE document activations, and probe training
    (registered for this purpose only; it does not check for cancellation).
  - A job record in state `running` blocks the delete too.
  - While the delete runs, any of these jobs trying to start fails with
    `JobIdReservedError` ("cannot start while its collection is being
    deleted"). Otherwise a job could write back what was just deleted.
- **Job records are matched by owner**, `config["collection_name"]`, which
  both record-writing jobs store; records without it fall back to their id.
  `poems_llm_labeling` is the LLM-labeling record of `poems`, even if a
  collection happens to have that name.
- **Uploads are removed only when safe:**
  - the file sits directly in `resources/uploads/`, never anywhere else (SAE
    parquet files under `resources/sae_vectors/` stay);
  - it was last modified before the dataset was created. Uploads overwrite
    same-named files, so a newer file is a re-upload waiting to be embedded;
  - no remaining dataset's `source_file`, no remaining ChromaDB collection's
    `source_file` metadata, and no string anywhere in a job record's config
    (the replayable input) resolves to it. An interrupted embed may still
    need it to resume. Paths are compared resolved, so non-ASCII names and
    `/var` versus `/private/var` spellings match.
- **Probe artifacts:** collection names are sanitised for paths, so
  `my-poems` and `my_poems` share `collections/my_poems/`. The folder is kept,
  with a warning in the log, while another collection maps to it. A name that
  sanitises to nothing never removes the artifacts root.
- **Order and failures:**
  1. ChromaDB first, then DuckDB. If the DuckDB step fails, the collection is
     still listed and deleting it again finishes the job.
  2. Once both stores are cleared, the deletion counts as done. Removing the
     probe folder, the job records and the upload is logged on failure and
     never fails the deletion; a retry would find nothing to delete.
  3. The mutation returns False when the collection exists in neither store
     or a store step fails.
  4. Everything runs in a worker thread, off the event loop.
- **Dataset order:** full-text index, then dataset row, then items table.
  - The index schema `fts_main_items_<dataset>` survives `DROP TABLE`.
  - DuckDB can reject the row delete (a foreign-key parent) while another
    read is open. The dataset then stays whole, with its items and listed
    under its own name, and its index is marked dirty so the next search
    rebuilds it.
- **Frontend:** `lib/utils/collectionDeletion.ts::runDeleteCollection`
  rejects with the server's message. `CollectionDetailPane` shows it in the
  delete dialog's error line, for example the running job to cancel. The
  dialog text lists what goes.

## Known gaps

- **Same-named uploads overwrite each other** (`API/upload.py`). The
  modified-time rule protects a re-upload, but two datasets embedded from
  successive versions of one name still share a path. Unique upload names
  would remove that ambiguity.
- **`_sanitize_table_name` collisions:** `my-poems` and `my_poems` datasets
  share one `items_my_poems` table, so deleting either drops the other's
  items. This is pre-existing and not fixed here.
- **Not removed:**
  - SAE features and activations (`sae_features`, `sae_activations`, keyed by
    model and SAE). They back the `/sae` explorer independently of any
    scatter collection.
  - Leftovers of collections deleted before 2026-10-19. On the live store
    that day there were 2 orphan job records (`sanskrit_dataset`,
    `sanskrit_dataset_dev`); probe folders had no orphans. 12 uploads
    (0.37 GB) were not referenced by any ChromaDB collection or job, but
    DuckDB was not checked, so they are not certainly orphans. Orphaned HNSW
    directories have their own prune script
    (`scripts/prune_chroma_orphans.py`).

## Tests

- `unit_tests/test_collection_deletion.py`: the service.
  - What goes: everything a collection owns, including an upload known only
    from ChromaDB metadata.
  - What stays: the shared dataset, a shared upload, an upload still needed
    by a job (ASCII and non-ASCII names), a re-uploaded file, a file outside
    the uploads folder, a shared probe folder, and the job records of another
    owner.
  - Failures and jobs: cleanup failures still report success; both kinds of
    running job block the delete; no job can start mid-delete; an unknown
    collection changes nothing.
- `unit_tests/test_delete_collection_mutation.py`: the resolver, with
  resources isolated in temporary folders, including the busy error.
- `unit_tests/test_duckdb_client.py::TestDeleteCollection`: DuckDB rows for
  every child table, the full-text index, the dataset-row ordering and its
  dirty flag.
- `unit_tests/test_probe_graphql.py::test_training_is_registered_as_a_running_job`
  and `unit_tests/test_job_ids.py`.
- `embedding_visualization/lib/utils/__tests__/collectionDeletion.test.ts`.
