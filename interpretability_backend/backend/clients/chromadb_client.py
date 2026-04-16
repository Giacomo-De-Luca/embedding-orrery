"""ChromaDB client wrapper with filtering and search capabilities."""

import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, Any, List
import numpy as np
from pathlib import Path
import json
import logging
import re

logger = logging.getLogger("star_map")

# Import embedding function factory to ensure correct model is used
from ..embedding_functions.create_embedding_function import create_embedding_function, get_device
from ..embedding_functions.config import EmbeddingModelConfig, EmbeddingProvider


# Gemini task type mapping: document task type -> query task type
# When documents are embedded with RETRIEVAL_DOCUMENT, queries should use RETRIEVAL_QUERY
GEMINI_QUERY_TASK_MAP = {
    "RETRIEVAL_DOCUMENT": "RETRIEVAL_QUERY",
    # Other task types don't have document/query variants, so they stay the same
}


def _map_gemini_task_type_for_query(task_type: Optional[str]) -> Optional[str]:
    """Map Gemini document task type to query task type.

    Args:
        task_type: The stored task type (used for document embedding)

    Returns:
        The appropriate task type for query embedding
    """
    if task_type is None:
        return None
    return GEMINI_QUERY_TASK_MAP.get(task_type, task_type)


class ChromaDBClient:
    """Wrapper for ChromaDB operations with filtering support."""

    def __init__(self, db_path: str = None):
        """Initialize ChromaDB client.

        Args:
            db_path: Path to ChromaDB persistent storage. If None, uses project root.
        """
        if db_path is None:
            # Go up from backend/clients/ to backend/ to interpretability/
            interpretability_root = Path(__file__).parent.parent.parent
            db_path = interpretability_root / "resources" / "vector_db"
        else:
            db_path = Path(db_path)

        self.db_path = db_path.resolve()
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all available collections.

        Returns:
            List of collection info dictionaries
        """
        collections = self.client.list_collections()
        return [
            {
                "name": col.name,
                "metadata": col.metadata,
                "count": col.count()
            }
            for col in collections
        ]

    def get_collection(
        self,
        name: str,
        load_embedding_function: bool = False,
        for_query: bool = False,
        query_prompt: Optional[str] = None
    ):
        """Get a collection by name.

        Args:
            name: Collection name
            load_embedding_function: If True, loads the embedding function for query operations.
                                     If False (default), returns collection without EF for read-only ops.
            for_query: If True, configures EF for query embedding (QWEN adds instruction prefix)
            query_prompt: Override prompt for query embedding (can be known name or custom string)

        Returns:
            ChromaDB collection (with or without embedding function)
        """
        try:
            if not load_embedding_function:
                # Fast path: read-only operations (projection data, get items)
                # No embedding function needed - just return collection
                return self.client.get_collection(name=name)

            # Slow path: load embedding function for semantic search with text
            collection = self.client.get_collection(name=name)

            # Check metadata for embedding model info
            metadata = collection.metadata or {}
            provider_str = metadata.get("embedding_provider")
            model_name = metadata.get("embedding_model")

            if provider_str and model_name:
                try:
                    # Construct config to create correct EF
                    provider = EmbeddingProvider(provider_str)

                    # Retrieve provider-specific params from metadata
                    task = metadata.get("embedding_task")  # QWEN: query instruction
                    task_type = metadata.get("embedding_task_type")  # Gemini: optimization type

                    # Retrieve SentenceTransformers prompt from metadata
                    # Use override if provided, otherwise use stored value
                    prompt = query_prompt or metadata.get("embedding_prompt")

                    # For Gemini, map document task type to query task type when searching
                    # e.g., RETRIEVAL_DOCUMENT -> RETRIEVAL_QUERY
                    if for_query and provider == EmbeddingProvider.GEMINI:
                        task_type = _map_gemini_task_type_for_query(task_type)

                    config = EmbeddingModelConfig(
                        provider=provider,
                        model_name=model_name,
                        task=task,
                        task_type=task_type,
                        prompt=prompt
                    )

                    # Get embedding dimension from metadata (avoid test embedding)
                    embedding_dim = metadata.get("embedding_dim")

                    # Create EF (autodetect device is safe for inference)
                    device = get_device()
                    ef, _ = create_embedding_function(
                        config,
                        device,
                        known_dimension=embedding_dim,
                        is_query=for_query  # QWEN: adds instruction prefix when True
                    )

                    # Re-get collection with specific EF
                    return self.client.get_collection(name=name, embedding_function=ef)

                except Exception as e:
                    # If we can't load the specific model (e.g. missing API key),
                    # fallback to the collection with default EF (will work for retrieval, fail for query)
                    print(f"Warning: Could not load embedding function for '{name}': {e}")
                    return collection

            return collection
        except Exception as e:
            raise ValueError(f"Collection '{name}' not found: {e}")

    def get_all_items(
        self,
        collection_name: str,
        limit: Optional[int] = None,
        offset: int = 0,
        where: Optional[Dict[str, Any]] = None,
        include: List[str] = ["metadatas", "documents", "embeddings"]
    ) -> Dict[str, Any]:
        """Get items from collection with filtering.

        Args:
            collection_name: Name of the collection
            limit: Maximum number of items to return
            offset: Number of items to skip
            where: ChromaDB where filter (e.g., {"pos": "n"})
            include: What to include in results

        Returns:
            Dictionary with ids, embeddings, metadatas, documents
        """
        # Read-only operation - no embedding function needed
        collection = self.get_collection(collection_name, load_embedding_function=False)

        # Get total count first
        if where:
            total = collection.count()  # Note: count() doesn't support where clause
        else:
            total = collection.count()

        # Calculate actual limit
        if limit is None:
            limit = total

        # ChromaDB get with filtering
        results = collection.get(
            limit=limit,
            offset=offset,
            where=where,
            include=include
        )

        return results

    def semantic_search(
        self,
        collection_name: str,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",  # cosine, l2, or ip
        query_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform semantic search on collection.

        Args:
            collection_name: Name of the collection
            query_texts: Text queries to embed and search
            query_embeddings: Pre-computed embedding vectors
            n_results: Number of results to return
            where: ChromaDB where filter
            distance_metric: Distance metric (cosine, l2, ip)
            query_prompt: Prompt to override for query embedding (can be known name or custom string)

        Returns:
            Query results with ids, distances, metadatas, documents
        """
        # Only load EF if using query_texts (not query_embeddings)
        needs_ef = query_texts is not None
        # When embedding text queries, use query mode (QWEN adds instruction prefix)
        collection = self.get_collection(
            collection_name,
            load_embedding_function=needs_ef,
            for_query=needs_ef,
            query_prompt=query_prompt
        )

        # Validate inputs
        if query_texts is None and query_embeddings is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")

        if query_texts is not None and query_embeddings is not None:
            raise ValueError("Provide either query_texts or query_embeddings, not both")

        # Perform query
        results = collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["metadatas", "documents", "distances", "embeddings"]
        )

        # Convert distances to similarities based on metric
        distances = results.get("distances", [[]])[0]
        if distance_metric == "cosine":
            # Cosine distance [0, 2] -> cosine similarity [-1, 1]
            similarities = [1 - d for d in distances]
        elif distance_metric == "l2":
            # L2 distance -> similarity (inverse)
            similarities = [1 / (1 + d) for d in distances]
        elif distance_metric == "ip":
            # Inner product (already similarity)
            similarities = [-d for d in distances]  # Negate to get positive similarity
        else:
            similarities = [1 - d for d in distances]  # Default to cosine

        # Add similarities to results
        results["similarities"] = [similarities]

        return results

    def get_projection_data(
        self,
        collection_name: str,
        projection_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get full projection data for visualization.

        This retrieves all items with projections (PCA/UMAP 2D and 3D) from ChromaDB metadata.
        Works with any data source - no hardcoded field names.

        Returns a generic structure:
        - ids: unique identifiers
        - documents: embedded text content
        - item_metadata: raw metadata per item (flexible schema)
        - available_fields: list of metadata field names
        - Projections: PCA and UMAP coordinates (only requested types are parsed)

        Args:
            collection_name: Name of the collection
            projection_types: Which projections to parse (e.g. ["umap_2d", "umap_3d"]).
                            None means all four. Non-requested projections return None.

        Returns:
            Dictionary with projection data and metadata
        """
        ALL_PROJECTION_TYPES = ("pca_2d", "pca_3d", "umap_2d", "umap_3d")
        requested = set(projection_types) if projection_types else set(ALL_PROJECTION_TYPES)

        # Read-only operation - no embedding function needed
        collection = self.get_collection(collection_name, load_embedding_function=False)

        # Get all items with metadata
        results = collection.get(
            include=["metadatas", "documents"]
        )

        ids = results["ids"]
        documents = results["documents"] or [""] * len(ids)
        raw_metadatas = results["metadatas"] or [{}] * len(ids)

        # Track available fields across all items
        available_fields = set()

        # Extract data — only parse requested projections
        item_metadata = []
        projections = {pt: [] for pt in requested}

        for metadata in raw_metadatas:
            # Track available fields (excluding projection fields)
            for key in metadata.keys():
                if key not in ALL_PROJECTION_TYPES:
                    available_fields.add(key)

            # Store raw metadata (excluding projection coordinates)
            clean_metadata = {k: v for k, v in metadata.items()
                             if k not in ALL_PROJECTION_TYPES}
            item_metadata.append(clean_metadata)

            # Extract only requested projection coordinates from metadata
            try:
                for pt in requested:
                    default = "[0, 0, 0]" if "3d" in pt else "[0, 0]"
                    projections[pt].append(json.loads(metadata.get(pt, default)))
            except (json.JSONDecodeError, TypeError):
                for pt in requested:
                    projections[pt].append([0, 0, 0] if "3d" in pt else [0, 0])

        # Get collection metadata
        collection_metadata = collection.metadata or {}

        # Parse variance arrays from JSON strings
        pca_2d_variance = None
        pca_3d_variance = None
        try:
            if "pca_2d_variance" in collection_metadata:
                pca_2d_variance = json.loads(collection_metadata["pca_2d_variance"])
            if "pca_3d_variance" in collection_metadata:
                pca_3d_variance = json.loads(collection_metadata["pca_3d_variance"])
        except (json.JSONDecodeError, TypeError):
            pass

        # Parse field_analysis from collection metadata if present
        field_analysis = None
        field_analysis_str = collection_metadata.get("field_analysis")
        if field_analysis_str:
            try:
                field_analysis = json.loads(field_analysis_str)
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "ids": ids,
            "documents": documents,
            "item_metadata": item_metadata,
            "available_fields": sorted(list(available_fields)),
            # Projections — None for non-requested types
            "pca_2d": projections.get("pca_2d"),
            "pca_3d": projections.get("pca_3d"),
            "umap_2d": projections.get("umap_2d"),
            "umap_3d": projections.get("umap_3d"),
            # Collection-level metadata
            "metadata": {
                "total_items": len(ids),
                "embedding_dim": collection_metadata.get("embedding_dim", 384),
                "embedding_provider": collection_metadata.get("embedding_provider"),
                "embedding_model": collection_metadata.get("embedding_model"),
                "timestamp": collection_metadata.get("timestamp", collection_metadata.get("created_at", "")),
                "pca_2d_variance": pca_2d_variance,
                "pca_3d_variance": pca_3d_variance,
                # Source info (from collection metadata)
                "source_dataset": collection_metadata.get("source_dataset"),
                "source_split": collection_metadata.get("source_split"),
                "source_file": collection_metadata.get("source_file"),
                "embedded_columns": collection_metadata.get("embedded_columns"),
                "has_projections": collection_metadata.get("has_projections", False),
                # Prompt info (for models like Gemma Embedding)
                "embedding_prompt": collection_metadata.get("embedding_prompt"),
                # Topic extraction metadata
                "has_topics": collection_metadata.get("has_topics", False),
                "topic_count": collection_metadata.get("topic_count"),
                "topics_extracted_at": collection_metadata.get("topics_extracted_at"),
                # Pre-computed field analysis (if available)
                "field_analysis": field_analysis,
            }
        }

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get detailed information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection info and metadata
        """
        # Read-only operation - no embedding function needed
        collection = self.get_collection(collection_name, load_embedding_function=False)
        metadata = collection.metadata or {}

        # Parse JSON fields in metadata
        parsed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, str) and value.startswith('['):
                try:
                    parsed_metadata[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed_metadata[key] = value
            else:
                parsed_metadata[key] = value

        return {
            "name": collection_name,
            "count": collection.count(),
            "metadata": parsed_metadata
        }

    def update_collection_metadata(
        self,
        collection_name: str,
        metadata_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update metadata for a collection.

        For each key in metadata_updates:
        - If the key exists in current metadata, it will be overwritten
        - If the key doesn't exist, it will be added
        - If the value is None, the key will be deleted from metadata

        Args:
            collection_name: Name of the collection to update
            metadata_updates: Dictionary of metadata key/value pairs to set.
                              Use None as value to delete a key.

        Returns:
            Dictionary with updated metadata
        """
        # Read-only operation initially (just metadata update) - no embedding function needed
        collection = self.get_collection(collection_name, load_embedding_function=False)
        current_metadata = collection.metadata or {}

        # Merge: existing metadata + updates (updates overwrite existing keys)
        merged_metadata = {**current_metadata, **metadata_updates}

        # Filter out None values (signals deletion)
        new_metadata = {k: v for k, v in merged_metadata.items() if v is not None}

        # ChromaDB's modify() updates the collection metadata
        collection.modify(metadata=new_metadata)

        return {
            "name": collection_name,
            "metadata": new_metadata
        }

    # ---- Text Search ----

    DOCUMENT_SENTINEL = "__document__"
    _SNIPPET_RADIUS = 100  # chars each side of the match

    def text_search(
        self,
        collection_name: str,
        query: str,
        fields: Optional[List[str]] = None,
        mode: str = "contains",
        case_sensitive: bool = False,
    ) -> Dict[str, Any]:
        """Full-text search across document content and/or metadata fields.

        Uses ChromaDB's native ``where_document`` for document text and falls
        back to a lightweight Python filter for metadata fields (ChromaDB does
        not support substring matching on metadata).

        Args:
            collection_name: Name of the collection to search.
            query: The search string.
            fields: List of fields to search.  Use ``"__document__"`` for the
                embedded document text.  ``None`` (default) searches documents
                only (fastest path).
            mode: ``"contains"`` for substring match, ``"exact"`` for full-value
                match.
            case_sensitive: When ``False`` (default) matching is
                case-insensitive.

        Returns:
            ``{"matches": [...], "total_matches": int}``
        """
        if not query:
            return {"matches": [], "total_matches": 0}

        collection = self.get_collection(collection_name, load_embedding_function=False)

        # Resolve target fields
        if fields is None:
            target_fields = [self.DOCUMENT_SENTINEL]
        else:
            target_fields = list(fields)

        search_document = self.DOCUMENT_SENTINEL in target_fields
        metadata_fields = [f for f in target_fields if f != self.DOCUMENT_SENTINEL]

        # Collect matches: id -> first matched field
        matched: Dict[str, str] = {}

        # --- 1. Document search (native ChromaDB) ---
        if search_document:
            where_doc = self._build_where_document(query, mode, case_sensitive)
            try:
                doc_results = collection.get(where_document=where_doc, include=[])
                for item_id in doc_results["ids"]:
                    matched.setdefault(item_id, self.DOCUMENT_SENTINEL)
            except Exception as e:
                logger.warning("Document search failed for query %r: %s", query, e)

        # --- 2. Metadata field search (Python filter) ---
        if metadata_fields:
            meta_results = collection.get(include=["metadatas"])
            ids = meta_results["ids"]
            metadatas = meta_results["metadatas"] or [{}] * len(ids)

            for item_id, meta in zip(ids, metadatas):
                if item_id in matched:
                    continue  # already matched via document
                for field in metadata_fields:
                    value = meta.get(field)
                    if value is None:
                        continue
                    str_val = str(value)
                    if self._matches(str_val, query, mode, case_sensitive):
                        matched[item_id] = field
                        break

        # --- 3. Build snippets for document matches ---
        doc_match_ids = [mid for mid, f in matched.items() if f == self.DOCUMENT_SENTINEL]
        snippets: Dict[str, str] = {}
        if doc_match_ids:
            docs_result = collection.get(ids=doc_match_ids, include=["documents"])
            for item_id, doc in zip(docs_result["ids"], docs_result["documents"] or []):
                if doc:
                    snippets[item_id] = self._extract_snippet(doc, query, case_sensitive)

        # --- 4. Assemble response ---
        matches = []
        for item_id, field in matched.items():
            matches.append({
                "id": item_id,
                "matched_field": field,
                "snippet": snippets.get(item_id),
            })

        return {"matches": matches, "total_matches": len(matches)}

    # -- helpers --

    @staticmethod
    def _build_where_document(query: str, mode: str, case_sensitive: bool) -> Dict[str, Any]:
        """Build a ChromaDB ``where_document`` clause."""
        escaped = re.escape(query)
        if mode == "exact":
            # Full-document match with anchors
            pattern = f"^{escaped}$"
            if not case_sensitive:
                pattern = f"(?i){pattern}"
            return {"$regex": pattern}
        else:  # contains
            if case_sensitive:
                return {"$contains": query}
            return {"$regex": f"(?i){escaped}"}

    @staticmethod
    def _matches(value: str, query: str, mode: str, case_sensitive: bool) -> bool:
        """Check if *value* matches *query* for metadata field filtering."""
        if case_sensitive:
            return (query == value) if mode == "exact" else (query in value)
        return (query.lower() == value.lower()) if mode == "exact" else (query.lower() in value.lower())

    @classmethod
    def _extract_snippet(cls, document: str, query: str, case_sensitive: bool) -> str:
        """Return a ~200-char context window around the first match."""
        doc_search = document if case_sensitive else document.lower()
        q = query if case_sensitive else query.lower()
        idx = doc_search.find(q)
        if idx == -1:
            return document[:cls._SNIPPET_RADIUS * 2]
        start = max(0, idx - cls._SNIPPET_RADIUS)
        end = min(len(document), idx + len(query) + cls._SNIPPET_RADIUS)
        snippet = document[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(document):
            snippet = snippet + "..."
        return snippet
