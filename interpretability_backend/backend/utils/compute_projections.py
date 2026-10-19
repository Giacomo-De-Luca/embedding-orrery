"""
Projection computation utilities.

Computes PCA and UMAP projections for embedding collections: embeddings are
read from ChromaDB (``embedding_loader.load_all_embeddings``), each projection is fitted by ``fit_projections`` and stored
in DuckDB as soon as it is ready.

The fits run through ``CancellableTask`` so a cancel request stops even a UMAP
that is already running: on large collections ``fit_projections`` runs in a
spawned child that is killed on cancel, and every DuckDB write stays in this
(parent) process. Projections stored before the cancel are kept.
"""

import importlib.util
import traceback
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..services.progress_emitter import emit_progress
from . import cancellable_task
from .cancellable_task import CancellableTask
from .duckdb_sync import _get_db as _get_duckdb
from .embedding_loader import load_all_embeddings

ProjectionKind = Literal["pca2d", "pca3d", "umap2d", "umap3d"]


@dataclass(frozen=True)
class ProjectionSpec:
    """One projection: how it is requested, fitted and stored."""

    kind: ProjectionKind
    stored_as: str  # projection_type in DuckDB
    method: Literal["pca", "umap"]
    n_components: int
    label: str

    def fit(self, embeddings: np.ndarray) -> tuple[np.ndarray, list[float] | None]:
        """Return ``(coords, explained variance ratio or None)``."""
        # Deferred: sklearn and umap stay out of the API's import graph
        # (lean-import boundary, see the backend CLAUDE.md).
        if self.method == "pca":
            from sklearn.decomposition import PCA

            pca = PCA(n_components=self.n_components, random_state=7)
            coords = pca.fit_transform(embeddings)
            return coords, pca.explained_variance_ratio_.tolist()

        import umap

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=7,
            verbose=False,
        )
        return reducer.fit_transform(embeddings), None


PROJECTION_SPECS: dict[str, ProjectionSpec] = {
    spec.kind: spec
    for spec in (
        ProjectionSpec("pca2d", "pca_2d", "pca", 2, "PCA 2D"),
        ProjectionSpec("pca3d", "pca_3d", "pca", 3, "PCA 3D"),
        ProjectionSpec("umap2d", "umap_2d", "umap", 2, "UMAP 2D"),
        ProjectionSpec("umap3d", "umap_3d", "umap", 3, "UMAP 3D"),
    )
}


def fit_projections(channel, embeddings: np.ndarray, kinds: list[str]) -> None:
    """Task target: fit each projection and send ``(kind, coords, variance)``
    as soon as it is done, so finished projections are stored even if a later
    one is cancelled."""
    for kind in kinds:
        coords, variance = PROJECTION_SPECS[kind].fit(embeddings)
        channel.send((kind, np.asarray(coords, dtype=np.float32), variance))


def compute_projections_for_collection(
    collection_name: str,
    projection_type: list[ProjectionKind] | None = None,
    job_id: str | None = None,
    cancel_event=None,
) -> bool:
    """Compute PCA and UMAP projections for a collection.

    Reads embeddings from ChromaDB, computes projections, stores in DuckDB.

    Args:
        collection_name: Name of the collection
        projection_type: Which projections to compute. None means all four.
        job_id: Optional job ID for WebSocket progress emission
        cancel_event: The job's cancel event; a cancel raises JobCancelled
            (projections stored before it are kept)

    Returns:
        True if successful, False otherwise
    """
    kinds = [k for k in (projection_type or PROJECTION_SPECS) if k in PROJECTION_SPECS]
    if importlib.util.find_spec("umap") is None:
        print("Warning: umap-learn not installed, skipping UMAP projections")
        kinds = [k for k in kinds if PROJECTION_SPECS[k].method != "umap"]

    try:
        # DuckDB: resolve vector_collection for storing projections
        duckdb = _get_duckdb()
        if not duckdb or not duckdb.get_vector_collection(collection_name):
            print(f"Warning: vector collection {collection_name!r} not found in DuckDB")
            return False

        print(f"Loading embeddings from {collection_name}...")
        ids, embeddings = load_all_embeddings(collection_name, cancel_event)
        print(f"Computing projections for {len(ids)} items...")

        completed = 0

        def store(item) -> None:
            nonlocal completed
            kind, coords, variance = item
            spec = PROJECTION_SPECS[kind]
            duckdb.insert_projections_batch(collection_name, ids, spec.stored_as, coords.tolist())
            duckdb.upsert_projection_metadata(collection_name, spec.stored_as, variance=variance)
            completed += 1
            print(f"{spec.label} complete")
            if job_id:
                emit_progress(
                    job_id=job_id,
                    status="running",
                    items_processed=completed,
                    total_items=len(kinds),
                    current_batch=0,
                    total_batches=0,
                    message=f"Projections: {completed}/{len(kinds)} complete ({spec.label} done)",
                )

        task = CancellableTask(
            cancel_event,
            isolate=cancellable_task.SETTINGS.should_isolate(len(ids)),
            on_item=store,
        )
        task.run(fit_projections, embeddings, kinds)
        print("Projections computed and stored in DuckDB.")
        return True

    except Exception as e:
        print(f"Error computing projections: {e}")
        traceback.print_exc()
        return False
