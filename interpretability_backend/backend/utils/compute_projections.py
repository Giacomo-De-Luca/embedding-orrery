"""
Projection computation utilities.

This module handles computing PCA and UMAP projections for embedding collections.
"""

import chromadb
from chromadb.config import Settings
import time
import json
import numpy as np
from tqdm import tqdm
from typing import Optional, Union, List, Literal

from ..embedding_functions.config import DB_PATH


def compute_projections_for_collection(collection_name: str, projection_type: Optional[List[Literal["pca2d", "pca3d", "umap2d", "umap3d"]]] = None) -> bool:
    """
    Compute PCA and UMAP projections for a collection.

    This is a simplified version that reuses logic from compute_projections.py.

    Args:
        collection_name: Name of the collection

    Returns:
        True if successful, False otherwise
    """
    try:
        from sklearn.decomposition import PCA

        # Check if umap is available
        try:
            import umap
            has_umap = True
        except ImportError:
            has_umap = False
            print("Warning: umap-learn not installed, skipping UMAP projections")

        db_path = str(DB_PATH.resolve())
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Don't load embedding function - we only need to read existing embeddings
        collection = client.get_collection(
            name=collection_name,
            embedding_function=None
        )

        # Get all embeddings in batches (metadata not needed yet)
        print(f"Loading embeddings from {collection_name}...")

        # First, get total count
        count = collection.count()
        print(f"Total items: {count}")

        # Load embeddings and IDs in batches
        load_batch_size = 5000
        embeddings_list = []
        ids_list = []

        for offset in tqdm(range(0, count, load_batch_size), desc="Loading embeddings"):
            limit = min(load_batch_size, count - offset)
            batch_data = collection.get(
                include=["embeddings"],  # Only embeddings, not metadata
                limit=limit,
                offset=offset
            )
            embeddings_list.extend(batch_data['embeddings'])
            ids_list.extend(batch_data['ids'])

        embeddings = np.array(embeddings_list)
        ids = ids_list
        del embeddings_list  # Free memory

        print(f"Computing projections for {len(ids)} items...")
        update_batch_size = 1000


        if projection_type is not None and "pca2d" in projection_type:

        # Compute and store PCA 2D
            print("Computing PCA 2D projections...")
            pca_2d = PCA(n_components=2, random_state=42)
            coords_pca_2d = pca_2d.fit_transform(embeddings)
            pca_2d_variance = pca_2d.explained_variance_ratio_.tolist()

            print("Storing PCA 2D projections...")
            for i in tqdm(range(0, len(ids), update_batch_size), desc="Updating PCA 2D"):
                batch_ids = ids[i:i + update_batch_size]
                batch_data = collection.get(ids=batch_ids, include=["metadatas"])
                batch_metadatas = []
                for j, idx in enumerate(range(i, min(i + update_batch_size, len(ids)))):
                    meta = batch_data['metadatas'][j].copy()
                    meta['pca_2d'] = json.dumps(coords_pca_2d[idx].tolist())
                    batch_metadatas.append(meta)
                collection.update(ids=batch_ids, metadatas=batch_metadatas)

            del coords_pca_2d  # Free memory
            print("PCA 2D complete, memory freed")

        if projection_type is not None and "pca3d" in projection_type:


            # Compute and store PCA 3D
            print("Computing PCA 3D projections...")
            pca_3d = PCA(n_components=3, random_state=42)
            coords_pca_3d = pca_3d.fit_transform(embeddings)
            pca_3d_variance = pca_3d.explained_variance_ratio_.tolist()

            print("Storing PCA 3D projections...")
            for i in tqdm(range(0, len(ids), update_batch_size), desc="Updating PCA 3D"):
                batch_ids = ids[i:i + update_batch_size]
                batch_data = collection.get(ids=batch_ids, include=["metadatas"])
                batch_metadatas = []
                for j, idx in enumerate(range(i, min(i + update_batch_size, len(ids)))):
                    meta = batch_data['metadatas'][j].copy()
                    meta['pca_3d'] = json.dumps(coords_pca_3d[idx].tolist())
                    batch_metadatas.append(meta)
                collection.update(ids=batch_ids, metadatas=batch_metadatas)

            del coords_pca_3d  # Free memory
            print("PCA 3D complete, memory freed")

        if projection_type is not None and "umap2d" in projection_type:


            # Compute and store UMAP if available
            if has_umap:
                print("Computing UMAP 2D projections...")
                reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                    metric='cosine', random_state=42, verbose=True)
                coords_umap_2d = reducer_2d.fit_transform(embeddings)

                print("Storing UMAP 2D projections...")
                for i in tqdm(range(0, len(ids), update_batch_size), desc="Updating UMAP 2D"):
                    batch_ids = ids[i:i + update_batch_size]
                    batch_data = collection.get(ids=batch_ids, include=["metadatas"])
                    batch_metadatas = []
                    for j, idx in enumerate(range(i, min(i + update_batch_size, len(ids)))):
                        meta = batch_data['metadatas'][j].copy()
                        meta['umap_2d'] = json.dumps(coords_umap_2d[idx].tolist())
                        batch_metadatas.append(meta)
                    collection.update(ids=batch_ids, metadatas=batch_metadatas)

                del coords_umap_2d  # Free memory
                print("UMAP 2D complete, memory freed")

        if projection_type is not None and "umap3d" in projection_type:
                
                print("Computing UMAP 3D projections...")
                reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                                    metric='cosine', random_state=42, verbose=True)
                coords_umap_3d = reducer_3d.fit_transform(embeddings)

                print("Storing UMAP 3D projections...")
                for i in tqdm(range(0, len(ids), update_batch_size), desc="Updating UMAP 3D"):
                    batch_ids = ids[i:i + update_batch_size]
                    batch_data = collection.get(ids=batch_ids, include=["metadatas"])
                    batch_metadatas = []
                    for j, idx in enumerate(range(i, min(i + update_batch_size, len(ids)))):
                        meta = batch_data['metadatas'][j].copy()
                        meta['umap_3d'] = json.dumps(coords_umap_3d[idx].tolist())
                        batch_metadatas.append(meta)
                    collection.update(ids=batch_ids, metadatas=batch_metadatas)

                del coords_umap_3d  # Free memory
                print("UMAP 3D complete, memory freed")

            # Free embeddings - no longer needed
                del embeddings
                print("Embeddings freed from memory")

        # Update collection metadata
        current_metadata = collection.metadata or {}

        
        current_metadata.update({
            'pca_2d_variance': json.dumps(pca_2d_variance) if projection_type is not None and "pca2d" in projection_type else None,
            'pca_3d_variance': json.dumps(pca_3d_variance) if projection_type is not None and "pca3d" in projection_type else None,
            'has_projections': True,
            'projections_computed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        collection.modify(metadata=current_metadata)

        print("Projections computed and stored successfully!")
        return True

    except Exception as e:
        print(f"Error computing projections: {e}")
        return False

def main():
    # Example usage
    collection_name = "lacan_sentences_gemini_document"
    compute_projections_for_collection(collection_name, projection_type=["umap3d"])

if __name__ == "__main__":
    main()