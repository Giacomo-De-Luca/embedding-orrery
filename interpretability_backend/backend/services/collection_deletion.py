"""Deleting a vector collection together with everything derived from it.

What one collection owns, and where it lives:

- its vectors and their HNSW index directory, in ChromaDB
  (``ChromaDBClient.delete_collection``);
- its projections, topics, SAE document activations and probes in DuckDB, and
  its dataset (items, full-text index) once no other collection uses that
  dataset (``DuckDBClient.delete_collection``);
- its probe artifacts (``probe_collection_dir``);
- its jobs' records in ``job_state.json`` (those whose config names it);
- the uploaded file its dataset was embedded from (``UPLOADS_DIR``), once no
  other dataset, collection or job refers to it.

No job of the collection may run meanwhile: deletion is refused while one
does, and none can start until it finishes (``cancel_registry``), since a job
would write back what was just deleted.
"""

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from chromadb.errors import NotFoundError

from ..clients.chromadb_client import ChromaDBClient
from ..clients.duckdb_client import DuckDBClient
from ..utils.resource_paths import PROBING_RESULTS_DIR, UPLOADS_DIR
from . import cancel_registry
from .job_ids import JobIds
from .job_state import JobState, JobStateService, JobStatus
from .probing_types import probe_collection_dir, sanitize_field_key

logger = logging.getLogger("orrery")


class CollectionBusyError(RuntimeError):
    """A job of the collection is still running."""


class CollectionDeletionService:
    """Deletes a collection from every store that holds part of it."""

    def __init__(
        self,
        duckdb: DuckDBClient,
        chroma: ChromaDBClient,
        job_states: JobStateService,
        *,
        probing_dir: Path | None = None,
        uploads_dir: Path | None = None,
    ):
        self.duckdb = duckdb
        self.chroma = chroma
        self.job_states = job_states
        self.probing_root = Path(probing_dir or PROBING_RESULTS_DIR)
        self.uploads_dir = Path(uploads_dir or UPLOADS_DIR).resolve()

    def delete(self, collection_name: str) -> bool:
        """Delete the collection and everything it owns.

        Returns False when the collection exists in neither store. Raises
        ``CollectionBusyError`` while one of its jobs runs. ChromaDB goes
        first: if the DuckDB step fails, the collection is still listed and
        deleting it again finishes the job. Once both stores are cleared the
        deletion counts as done; failures removing what is left (files, job
        records) are logged.
        """
        try:
            with cancel_registry.reserved_job_ids(JobIds.for_collection(collection_name)):
                return self._delete(collection_name)
        except (cancel_registry.JobIdsActiveError, cancel_registry.JobIdReservedError) as error:
            raise CollectionBusyError(self._busy_message(collection_name, str(error))) from error

    def _delete(self, collection_name: str) -> bool:
        records = self._job_records(collection_name)
        running = [job.collection_name for job in records if job.status is JobStatus.RUNNING]
        if running:
            raise CollectionBusyError(self._busy_message(collection_name, ", ".join(running)))

        dataset_name = self.duckdb.get_dataset_name_for_collection(collection_name)
        dataset_name = dataset_name or collection_name
        dataset = self.duckdb.get_dataset(dataset_name)
        source_file = self._source_file(collection_name, dataset)

        try:
            self.chroma.delete_collection(collection_name)
            deleted_vectors = True
        except NotFoundError:
            deleted_vectors = False
        if not self.duckdb.delete_collection(collection_name) and not deleted_vectors:
            return False

        self._logged(self._remove_probe_artifacts, collection_name)
        for job in records:
            self._logged(self.job_states.remove_job, job.collection_name)
        if source_file and self.duckdb.get_dataset(dataset_name) is None:
            created_at = dataset["created_at"] if dataset else None
            self._logged(self._remove_upload_if_unused, Path(source_file), created_at)
        logger.info("Deleted collection %r and everything it owned", collection_name)
        return True

    @staticmethod
    def _busy_message(collection_name: str, job_ids: str) -> str:
        return (
            f"Collection '{collection_name}' has a running job ({job_ids}); "
            "cancel it before deleting the collection"
        )

    @staticmethod
    def _logged(step, *args) -> None:
        """Run a cleanup step; its failure is logged, not raised."""
        try:
            step(*args)
        except Exception:
            logger.exception("Collection cleanup step %s failed", step.__name__)

    def _job_records(self, collection_name: str) -> list[JobState]:
        """Job records owned by the collection.

        Matched by the owner the pipelines store in ``config["collection_name"]``,
        not by id: ``poems_llm_labeling`` is both the LLM-labeling id of
        ``poems`` and a possible collection name. Records without it fall back
        to their id.
        """
        return [
            job
            for job in self.job_states.list_jobs()
            if (job.config or {}).get("collection_name", job.collection_name) == collection_name
        ]

    def _source_file(self, collection_name: str, dataset: dict[str, Any] | None) -> str | None:
        """The local file the collection's dataset was embedded from, if any."""
        if dataset and dataset.get("source_file"):
            return dataset["source_file"]
        try:  # collections predating the DuckDB dataset registry
            collection = self.chroma.client.get_collection(collection_name, embedding_function=None)
        except NotFoundError:
            return None
        return (collection.metadata or {}).get("source_file")

    def _remove_probe_artifacts(self, collection_name: str) -> None:
        if not sanitize_field_key(collection_name):
            return  # would be the root of every collection's artifacts
        directory = probe_collection_dir(collection_name, self.probing_root)
        if not directory.is_dir():
            return
        others = self._collection_names()
        if any(probe_collection_dir(name, self.probing_root) == directory for name in others):
            logger.warning(
                "Kept %s: another collection's probes are stored under the same name", directory
            )
            return
        shutil.rmtree(directory)

    def _collection_names(self) -> set[str]:
        """Every collection still in either store."""
        chroma_names = {collection.name for collection in self.chroma.client.list_collections()}
        return chroma_names | set(self.duckdb.list_vector_collection_names())

    def _remove_upload_if_unused(self, source_file: Path, dataset_created_at) -> None:
        """Remove the upload the deleted dataset was embedded from, when safe.

        Only a file directly in the uploads folder; not one modified after the
        dataset was created (uploads overwrite same-named files, so that is a
        re-upload waiting to be embedded); not one any remaining dataset,
        collection or job record refers to.
        """
        path = source_file.resolve()
        if path.parent != self.uploads_dir or not path.is_file():
            return
        if dataset_created_at is not None and path.stat().st_mtime > dataset_created_at.timestamp():
            return
        if self._upload_in_use(path):
            return
        path.unlink()

    def _upload_in_use(self, path: Path) -> bool:
        sources = set(self.duckdb.dataset_source_files())
        for collection in self.chroma.client.list_collections():
            source = (collection.metadata or {}).get("source_file")
            if source:
                sources.add(source)
        for job in self.job_states.list_jobs():
            sources.update(_strings(job.config))  # anywhere in its replayable input
        return any(_same_file(source, path) for source in sources)


def _strings(value: Any) -> Iterator[str]:
    """Every string inside nested dicts and lists."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)


def _same_file(candidate: str, path: Path) -> bool:
    try:
        return Path(candidate).resolve() == path
    except (OSError, ValueError):
        return False
