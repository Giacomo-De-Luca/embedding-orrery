"""
Projection computation utilities.

Computes PCA and UMAP projections for embedding collections.
Reads embeddings from ChromaDB, stores projections in DuckDB.
"""

import chromadb
from chromadb.config import Settings
import time
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Literal

from ..embedding_functions.config import DB_PATH
from ..services.progress_emitter import emit_progress
from .duckdb_sync import _get_db as _get_duckdb


def compute_projections_for_collection(
    collection_name: str,
    projection_type: Optional[List[Literal["pca2d", "pca3d", "umap2d", "umap3d"]]] = None,
    job_id: Optional[str] = None,
) -> bool:
    """Compute PCA and UMAP projections for a collection.

    Reads embeddings from ChromaDB, computes projections, stores in DuckDB.

    Args:
        collection_name: Name of the collection
        projection_type: Which projections to compute. None means all four.
        job_id: Optional job ID for WebSocket progress emission

    Returns:
        True if successful, False otherwise
    """
    if projection_type is None:
        projection_type = ["pca2d", "pca3d", "umap2d", "umap3d"]

    try:
        from sklearn.decomposition import PCA

        try:
            import umap
            has_umap = True
        except ImportError:
            has_umap = False
            print("Warning: umap-learn not installed, skipping UMAP projections")

        # ChromaDB: read embeddings
        db_path = str(DB_PATH.resolve())
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name, embedding_function=None)

        # DuckDB: resolve vector_collection for storing projections
        duckdb = _get_duckdb()
        vc = duckdb.get_vector_collection_by_name(collection_name) if duckdb else None
        if not vc:
            print(f"Warning: vector collection {collection_name!r} not found in DuckDB")
            return False
        vc_id = vc["id"]
        ds_id = vc["dataset_id"]

        # Load embeddings in batches from ChromaDB
        print(f"Loading embeddings from {collection_name}...")
        count = collection.count()
        print(f"Total items: {count}")

        load_batch_size = 5000
        embeddings_list = []
        ids_list = []

        for offset in tqdm(range(0, count, load_batch_size), desc="Loading embeddings"):
            limit = min(load_batch_size, count - offset)
            batch_data = collection.get(
                include=["embeddings"],
                limit=limit,
                offset=offset,
            )
            embeddings_list.extend(batch_data["embeddings"])
            ids_list.extend(batch_data["ids"])

        embeddings = np.array(embeddings_list)
        ids = ids_list
        del embeddings_list

        print(f"Computing projections for {len(ids)} items...")

        # Progress tracking
        requested = [p for p in projection_type if p in ("pca2d", "pca3d", "umap2d", "umap3d")]
        total_projections = len(requested)
        completed_projections = 0

        def _emit_done(name: str):
            nonlocal completed_projections
            completed_projections += 1
            if job_id:
                emit_progress(
                    job_id=job_id, status="running",
                    items_processed=completed_projections, total_items=total_projections,
                    current_batch=0, total_batches=0,
                    message=f"Projections: {completed_projections}/{total_projections} complete ({name} done)",
                )

        pca_2d_variance = None
        pca_3d_variance = None

        # ---- PCA 2D ----
        if "pca2d" in projection_type:
            print("Computing PCA 2D projections...")
            pca_2d = PCA(n_components=2, random_state=42)
            coords = pca_2d.fit_transform(embeddings)
            pca_2d_variance = pca_2d.explained_variance_ratio_.tolist()

            duckdb.insert_projections_batch(vc_id, ds_id, ids, "pca_2d", coords.tolist())
            duckdb.upsert_projection_metadata(vc_id, "pca_2d", variance=pca_2d_variance)
            del coords
            print("PCA 2D complete")
            _emit_done("PCA 2D")

        # ---- PCA 3D ----
        if "pca3d" in projection_type:
            print("Computing PCA 3D projections...")
            pca_3d = PCA(n_components=3, random_state=42)
            coords = pca_3d.fit_transform(embeddings)
            pca_3d_variance = pca_3d.explained_variance_ratio_.tolist()

            duckdb.insert_projections_batch(vc_id, ds_id, ids, "pca_3d", coords.tolist())
            duckdb.upsert_projection_metadata(vc_id, "pca_3d", variance=pca_3d_variance)
            del coords
            print("PCA 3D complete")
            _emit_done("PCA 3D")

        # ---- UMAP 2D ----
        if "umap2d" in projection_type and has_umap:
            print("Computing UMAP 2D projections...")
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                metric="cosine", random_state=42, verbose=False)
            coords = reducer.fit_transform(embeddings)

            duckdb.insert_projections_batch(vc_id, ds_id, ids, "umap_2d", coords.tolist())
            duckdb.upsert_projection_metadata(vc_id, "umap_2d")
            del coords
            print("UMAP 2D complete")
            _emit_done("UMAP 2D")

        # ---- UMAP 3D ----
        if "umap3d" in projection_type and has_umap:
            print("Computing UMAP 3D projections...")
            reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                                metric="cosine", random_state=42, verbose=False)
            coords = reducer.fit_transform(embeddings)

            duckdb.insert_projections_batch(vc_id, ds_id, ids, "umap_3d", coords.tolist())
            duckdb.upsert_projection_metadata(vc_id, "umap_3d")
            del coords
            print("UMAP 3D complete")
            _emit_done("UMAP 3D")

        del embeddings
        print("Projections computed and stored in DuckDB.")
        return True

    except Exception as e:
        print(f"Error computing projections: {e}")
        import traceback
        traceback.print_exc()
        return False
