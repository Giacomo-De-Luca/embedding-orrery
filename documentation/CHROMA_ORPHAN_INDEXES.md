# Orphaned ChromaDB index directories

## Problem

A persistent ChromaDB store keeps each collection's HNSW vector index in its
own directory, `resources/vector_db/<vector segment id>/`, holding
`header.bin`, `data_level0.bin`, `length.bin`, `link_lists.bin` and sometimes
`index_metadata.pickle`. chromadb writes that directory on a collection's
first add.

`client.delete_collection(name)` removes the collection's catalog rows from
`chroma.sqlite3`, but it **never removes the directory**. The vectors stay on
disk with nothing referencing them, and this can never be undone by chromadb
itself.

Every delete leaked this way:
- `deleteCollection` in the UI
- overwriting a collection on a fresh, non-resumed embed with the same name
- re-ingesting SAE explanation vectors

By 2026-10-19 the live store held **65 orphan directories, 3.22 GB**,
alongside 71 live ones. The oldest dated from January 2026. They were pruned
that day with the backend stopped, and the store went from 10 GB to 7.2 GB.
All 71 live directories were still present afterwards, and self-queries on
sample collections returned each item as its own nearest neighbour.

## Upstream status

- Reported as chroma-core/chroma [#5259](https://github.com/chroma-core/chroma/issues/5259)
  (1.0.15+) and [#7853](https://github.com/chroma-core/chroma/issues/7853) (1.5.9).
- Fixed on chromadb `main` by [#7767](https://github.com/chroma-core/chroma/pull/7767)
  (merged 2026-10-09). It is not in a release yet; the latest is 1.5.9. This
  repo requires `chromadb>=1.3.4` and is locked to 1.3.4 in `uv.lock`.
- The upstream fix only cleans up deletions it recorded itself, so directories
  leaked before an upgrade stay on disk regardless.

## Fix in this repo

`backend/utils/chroma_segments.py::ChromaSegmentJanitor`:

- **`delete_collection(client, name)`** (classmethod):
  1. Reads the collection's VECTOR segment id from the store's catalog
     (read-only sqlite, since chromadb has no public API for it).
  2. Deletes the collection.
  3. Removes its directory.
  - It raises chromadb's `NotFoundError` for a missing collection.
  - It does nothing extra for in-memory clients.
  - If the catalog can't be read, or the directory can't be removed, it logs a
    warning and the directory is left for the prune. The collection is deleted
    either way.
  - A directory holding anything but HNSW files is left alone and logged.
  - `ChromaDBClient.delete_collection(name)` wraps it. The `deleteCollection`
    mutation runs it in a worker thread.
- **`orphan_dirs(min_age_seconds)` / `prune_orphans(min_age_seconds, dry_run)`**
  find and remove directories that no segment references.
  - **Ordering:** directories are listed *before* the catalog is read.
    chromadb registers segments at collection creation and writes the
    directory later, so a collection created meanwhile is never taken for an
    orphan.
  - **Age margin:** directories modified within the minimum age are skipped.
  - **Strict matching:** only real directories (no symlinks) named by a
    canonical, lowercase segment UUID and holding nothing but HNSW files
    qualify.
  - **No catalog, no prune:** a missing catalog raises instead of treating
    every directory as an orphan.
  - **Layout check:** it raises if the catalog has vector segments but no
    directory matches any of them. A chromadb upgrade that renamed index
    directories would otherwise make every live index look orphaned.
  - **Partial failures:** a directory that can't be removed is reported in
    `PruneResult.failed` and the rest are still removed.

Delete paths using it:

| Path | Status |
|---|---|
| `API/mutations.py` `deleteCollection` | `CollectionDeletionService` (in a worker thread) → `ChromaDBClient.delete_collection`, then `DuckDBClient.delete_collection` (see `COLLECTION_DELETION.md`) |
| `embedding_functions/ingest_sae.py` (replaces the explanation collection) | uses `ChromaDBClient.delete_collection` |
| `embedding_functions/embedding_job.py` `open_embedding_collection` (overwrite on a fresh embed) | uses `ChromaSegmentJanitor.delete_collection(client, name)` |

No other code path calls chromadb's `delete_collection`.

Tests: `unit_tests/test_chroma_segments.py`.
- They pin the upstream bug. If that test starts failing after a chromadb
  upgrade, chromadb removes directories itself and only the prune is still
  needed.
- Each test that deletes something checks that the *kept* collection's own
  directory survives, and that its index still answers a query correctly after
  a reopen. A lost index directory still reports the right `count()` but
  returns wrong neighbours, so a count check would not catch it.
- They also cover:
  - deletion with small and default index settings, empty collections and
    in-memory clients
  - reuse and reopen after deletion
  - an unreadable catalog and failed removals
  - orphan matching rules: non-UUID, uppercase UUID, symlink, foreign file,
    nested directory
  - a missing catalog, the layout check and the age margin
  - dry run versus real prune
  - script config validation (`dry_run = 0` is rejected, not read as false)
    and a nonzero exit on failures

**In-process caveat:** the catalog lookup opens `chroma.sqlite3` with
Python's own SQLite while chromadb's bundled SQLite has it open in the same
process. Closing that read-only connection releases the process's POSIX locks
on the file ([SQLite: two copies in one process](https://sqlite.org/howtocorrupt.html)).
Inside the backend that is harmless, since chromadb is the only writer. It only
matters if a second process uses the store while the backend runs, which
chromadb's persistent client does not support anyway. Run the prune with the
backend stopped.

## Reclaiming leaked space

`scripts/prune_chroma_orphans.py` with `scripts/prune_chroma_orphans_config.toml`:

| Key | Default | Meaning |
|---|---|---|
| `dry_run` | `true` | Only list orphans and their sizes (must be a real boolean) |
| `min_age_minutes` | `10` | Skip recently modified directories (integer ≥ 0) |
| `chroma_path` | empty (the live store) | Store to clean |

Unknown keys and wrong types are rejected. The script exits 1 if any
directory could not be removed.

```bash
# backend stopped; dry run lists what would go
uv run python -m interpretability_backend.scripts.prune_chroma_orphans
# then set dry_run = false and run again
```

The dry run's list on the live store matched an independent check: every
directory not named by a VECTOR segment, with none of them referenced by any
segment.

Removing the directories frees their disk space immediately.
`chroma.sqlite3` itself never shrinks after deletions (freed pages are only
reused); a `VACUUM` with the backend stopped would compact it.

## Related fixes (2026-10-19)

Deleting a collection now removes everything derived from it, not just the
vectors and rows below. See `COLLECTION_DELETION.md`.

- **`deleteCollection` deleted the wrong DuckDB rows and always reported
  success** (audit `AUDIT_2026-07-02.md`, HIGH). It passed the collection name
  to `delete_dataset` and ignored the result:
  - Deleting a re-embedded collection (dataset name ≠ collection name) left
    its DuckDB rows behind.
  - Deleting a source collection wiped every sibling collection's rows along
    with the dataset.

  The fix:
  - `DuckDBClient.delete_vector_collection(name)` removes one collection's
    rows: topics, projections, SAE document activations, probes and its
    registry row. `delete_dataset` now reuses it for each of its collections.
  - `DuckDBClient.delete_collection(name)` removes that collection, plus the
    dataset and items once no collection uses them. Legacy datasets with no
    registered collection are deleted by name.
  - The mutation deletes the vectors first, then the rows, both in worker
    threads. If the DuckDB step fails, the collection stays listed and
    deleting it again finishes the job.
  - When the last collection's dataset goes, its row is deleted before its
    items table is dropped. DuckDB can reject that foreign-key parent delete
    while another read is open, and a failure then leaves the dataset whole,
    deletable by its own name, rather than a dataset row without items.
  - It returns False when the collection exists in neither store or a step
    fails, and logs the error.
  - Tests: `unit_tests/test_duckdb_client.py::TestDeleteCollection`,
    `unit_tests/test_delete_collection_mutation.py`.
- **Collections created without an embedding function recorded chromadb's
  default 384-d embedder.** The default parameter value of
  `create_collection(embedding_function=…)` stamps
  `{"type": "known", "name": "default"}`. `ingest_sae` and
  `open_embedding_collection` (image and vector paths) now pass
  `embedding_function=None` explicitly, so nothing is recorded. Tests:
  `unit_tests/test_ingest_sae_vectors.py`, `unit_tests/test_embedding_job.py`.
- **Overwriting a collection with a different embedding model failed.**
  `open_embedding_collection` checked for an existing collection by attaching
  the *new* embedding function, and chromadb rejects one that differs from
  the collection's. The check now attaches none; the returned handle only ever
  receives explicit `embeddings`. Test:
  `unit_tests/test_embedding_job.py::TestOpenEmbeddingCollection::test_existence_check_attaches_no_embedding_function`.
