"""GraphQL query resolvers for embedding visualization backend."""

import asyncio
import json

import strawberry

# Import clients at module level
from ..clients.huggingface_client import (
    get_dataset_info as hf_get_info,
    get_dataset_preview as hf_get_preview,
)
from ..clients.local_data_client import (
    get_local_file_info as get_local_info,
    get_local_file_preview as get_local_preview,
)
from ..services.job_state import JobStatus, get_job_state_service
from ..services.probing_types import score_field_names as probe_score_field_names
from .chromadb_instance import get_chromadb_client
from .duckdb_instance import get_duckdb_client
from .interpret_instance import get_interpret_service
from .types import (
    # Chat history types
    ChatSessionDetail,
    ChatSessionInfo,
    ChatSessionMessage,
    Collection,
    CollectionMetadata,
    # Probe types
    CollectionProbesResult,
    DocumentActivationResult,
    DocumentActivationSearchResponse,
    DocumentRankingMode,
    EmbeddingItem,
    EmbeddingJob,
    ExtractTopicsResult,
    FilterInput,
    HFConfigInfo,
    HFDatasetInfo,
    HFDatasetPreview,
    HFFeatureInfo,
    HFSplitInfo,
    LocalFileInfo,
    LocalFilePreview,
    # Interpret types
    ModelStatus,
    ProbeInfo,
    ProbeScores,
    ProjectionData,
    SaeActivation,
    SaeActivationQuantileGroup,
    # SAE types
    SaeFeature,
    SaeFeatureSearchResult,
    SaeLogitEntry,
    SaeModelInfo,
    SemanticSearchResult,
    SimilarityMeasure,
    TextSearchMatch,
    TextSearchMode,
    TextSearchResponse,
    TopicInfo,
    TopicKeyword,
)


def _job_model_name(embedding_model) -> str | None:
    """Model name from a job-config ``embedding_model`` entry.

    HF/local jobs store a dict ``{provider, model_name, ...}`` (via
    ``_build_job_config``); re-embed jobs historically stored a bare string.
    Tolerate both so one malformed record can't null the whole jobs list.
    """
    if isinstance(embedding_model, dict):
        return embedding_model.get("model_name")
    if isinstance(embedding_model, str):
        return embedding_model
    return None


def _resolve_dataset_name(db, collection_name: str) -> str:
    """Resolve a collection_name to its parent dataset_name.

    For decoupled collections (where collection_name != dataset_name),
    looks up the vector_collections table. Falls back to collection_name
    if no vector_collection record exists (backward compatible).
    """
    vc = db.get_vector_collection(collection_name)
    if vc:
        return vc["dataset_name"]
    return collection_name


def _dicts_to_topic_infos(topic_dicts: list) -> list:
    """Convert DuckDB topic dicts to Strawberry TopicInfo list."""
    topics = []
    for t in topic_dicts:
        kw_list = t.get("keywords") or []
        keywords = [TopicKeyword(word=kw["word"], score=kw["score"]) for kw in kw_list]
        topics.append(
            TopicInfo(
                topic_id=t["topic_id"],
                keywords=keywords,
                label=t.get("label"),
                count=t.get("count", 0),
                subtopics=t.get("subtopics"),
            )
        )
    return topics


@strawberry.type
class Query:
    """GraphQL query root."""

    @strawberry.field
    def collections(self, info) -> list[Collection]:
        """List all available collections.

        Returns:
            List of collections with metadata
        """
        db = get_duckdb_client()
        datasets = db.list_datasets()

        return [
            Collection(name=ds["name"], metadata=ds["metadata"], count=ds["count"])
            for ds in datasets
        ]

    @strawberry.field
    def embedding_jobs(self, status: str | None = None, info=None) -> list[EmbeddingJob]:
        """List embedding jobs with their progress and configuration.

        Args:
            status: Optional filter - "running", "interrupted", or "completed"

        Returns:
            List of embedding jobs with progress information
        """
        job_service = get_job_state_service()

        # Convert string status to JobStatus enum if provided
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status)
            except ValueError:
                pass  # Invalid status, return all jobs

        jobs = job_service.list_jobs(status=status_filter)

        return [
            EmbeddingJob(
                collection_name=job.collection_name,
                status=job.status.value,
                job_type=job.job_type,
                # Progress from job state
                items_embedded=job.items_embedded,
                total_expected=job.total_expected,
                batches_completed=job.batches_completed,
                total_batches=job.total_batches,
                percent_complete=job.percent_complete,
                # Config summary for display
                source=job.source,
                columns=job.config.get("columns"),
                embedding_model=_job_model_name(job.config.get("embedding_model")),
                batch_size=job.config.get("batch_size", 100),
                started_at=job.started_at,
                # Full config for resume verification
                config=job.config,
            )
            for job in jobs
        ]

    @strawberry.field
    def huggingface_dataset_info(self, dataset_id: str, info=None) -> HFDatasetInfo:
        """Get information about a HuggingFace dataset.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., "squad", "glue")

        Returns:
            Dataset info with configs, splits, features, and metadata
        """
        result = hf_get_info(dataset_id)

        # Convert dataclass to GraphQL type
        configs = []
        for cfg in result.configs:
            splits = [
                HFSplitInfo(name=s.name, num_rows=s.num_rows, num_bytes=s.num_bytes)
                for s in cfg.splits
            ]
            features = [
                HFFeatureInfo(name=f.name, dtype=f.dtype, description=f.description)
                for f in cfg.features
            ]
            configs.append(HFConfigInfo(name=cfg.name, splits=splits, features=features))

        return HFDatasetInfo(
            dataset_id=result.dataset_id,
            description=result.description,
            license=result.license,
            configs=configs,
            default_config=result.default_config,
            error=result.error,
        )

    @strawberry.field
    def huggingface_dataset_preview(
        self,
        dataset_id: str,
        config: str | None = None,
        split: str = "train",
        n_rows: int = 5,
        info=None,
    ) -> HFDatasetPreview:
        """Get preview rows from a HuggingFace dataset.

        Args:
            dataset_id: HuggingFace dataset ID
            config: Configuration name (None for default)
            split: Split name (default: "train")
            n_rows: Number of rows to preview (default: 5)

        Returns:
            Preview with column names and sample rows
        """
        result = hf_get_preview(dataset_id, config, split, n_rows)

        return HFDatasetPreview(
            dataset_id=result.dataset_id,
            config=result.config,
            split=result.split,
            columns=result.columns,
            rows=result.rows,
            total_rows=result.total_rows,
            error=result.error,
        )

    @strawberry.field
    def local_file_info(self, file_path: str, info=None) -> LocalFileInfo:
        """Get information about a local data file.

        Args:
            file_path: Path to local file (parquet, json, csv)

        Returns:
            File info with columns, row count, and file metadata
        """
        print(f"DEBUG: local_file_info resolver called with file_path={file_path}")

        try:
            result = get_local_info(file_path)
            print(f"DEBUG: get_info returned: {result}")
        except Exception as e:
            print(f"DEBUG: get_info raised exception: {e}")
            raise e

        return LocalFileInfo(
            file_path=result.file_path,
            file_type=result.file_type,
            columns=result.columns,
            num_rows=result.num_rows,
            file_size_bytes=result.file_size_bytes,
            error=result.error,
        )

    @strawberry.field
    def local_file_preview(self, file_path: str, n_rows: int = 5, info=None) -> LocalFilePreview:
        """Get preview rows from a local data file.

        Args:
            file_path: Path to local file
            n_rows: Number of rows to preview (default: 5)

        Returns:
            Preview with column names and sample rows
        """
        result = get_local_preview(file_path, n_rows)

        return LocalFilePreview(
            file_path=result.file_path,
            columns=result.columns,
            rows=result.rows,
            total_rows=result.total_rows,
            error=result.error,
        )

    @strawberry.field
    def collection(
        self,
        name: str,
        info,
        projection_types: list[str] | None = None,
    ) -> ProjectionData | None:
        """Get complete projection data for a collection.

        Args:
            name: Collection name
            projection_types: Which projections to load (e.g. ["umap_2d", "umap_3d"]).
                            None means all four. Non-requested projections return null.

        Returns:
            Projection data with PCA/UMAP projections.
            Generic structure - no hardcoded field names.
        """
        db = get_duckdb_client()

        all_types = ["pca_2d", "pca_3d", "umap_2d", "umap_3d"]
        requested = projection_types or all_types

        # Load one projection type at a time from DuckDB
        # First pass: load items + first requested type to get shared data
        projections = {}
        items_data = None

        for ptype in requested:
            data = db.get_projection_data(name, ptype)
            if data is not None:
                projections[ptype] = data["coordinates"]
                if items_data is None:
                    items_data = data  # capture ids, documents, metadata from first result

        # If no projections found, try loading items without projections
        if items_data is None:
            dataset_name = _resolve_dataset_name(db, name)
            ds = db.get_dataset(dataset_name)
            if ds is None:
                return None
            # Load items directly
            rows = db.get_items_columns(dataset_name, ("id", "document", "metadata"))
            if not rows:
                return None
            import json as json_mod

            items_data = {
                "ids": [r[0] for r in rows],
                "documents": [r[1] for r in rows],
                "item_metadata": [
                    json_mod.loads(r[2]) if isinstance(r[2], str) and r[2] else {} for r in rows
                ],
                "available_fields": [],
                "metadata": {},
            }
            if items_data["item_metadata"]:
                all_keys = set()
                for m in items_data["item_metadata"]:
                    all_keys.update(m.keys())
                items_data["available_fields"] = sorted(all_keys)

        # Build collection-level metadata
        metadata = dict(items_data.get("metadata", {}))

        # Load topic data from DuckDB
        topic_data = db.get_active_topics(name)
        if topic_data and topic_data.get("topics"):
            metadata["has_topics"] = True
            metadata["topic_count"] = topic_data.get("topic_count") or len(
                [t for t in topic_data["topics"] if t["topic_id"] != -1]
            )
            metadata["topics_extracted_at"] = str(topic_data.get("extracted_at", ""))

            metadata["topics"] = _dicts_to_topic_infos(topic_data["topics"])

            # Load topic hierarchy from extraction
            if topic_data.get("topic_hierarchy"):
                import json as json_mod

                raw = topic_data["topic_hierarchy"]
                metadata["topic_hierarchy"] = json_mod.loads(raw) if isinstance(raw, str) else raw

        # Merge topic assignments into item_metadata so frontend sees topic_id/topic_label
        if topic_data and topic_data.get("topics"):
            ext_id = topic_data["id"]
            assignments = db.get_topic_assignments_raw(
                ext_id,
                ["item_id", "topic_id", "topic_label", "subtopic_id", "subtopic_label"],
            )
            assign_map = {}
            for row in assignments:
                assign_map[row[0]] = {
                    "topic_id": str(row[1]),
                    "topic_label": row[2] or "Unclustered",
                }
                if row[3] is not None:
                    assign_map[row[0]]["subtopic_id"] = str(row[3])
                if row[4] is not None:
                    assign_map[row[0]]["subtopic_label"] = row[4]

            for i, item_id in enumerate(items_data["ids"]):
                if item_id in assign_map:
                    items_data["item_metadata"][i].update(assign_map[item_id])

            # Add topic fields to available_fields
            avail = set(items_data["available_fields"])
            avail.update(["topic_id", "topic_label"])
            if any("subtopic_id" in a for a in assign_map.values()):
                avail.update(["subtopic_id", "subtopic_label"])
            items_data["available_fields"] = sorted(avail)

        return ProjectionData(
            ids=items_data["ids"],
            documents=items_data["documents"],
            item_metadata=items_data["item_metadata"],
            available_fields=items_data["available_fields"],
            pca_2d=projections.get("pca_2d"),
            pca_3d=projections.get("pca_3d"),
            umap_2d=projections.get("umap_2d"),
            umap_3d=projections.get("umap_3d"),
            metadata=CollectionMetadata(**metadata),
        )

    @strawberry.field
    def collection_topics(self, collection_name: str, info=None) -> ExtractTopicsResult | None:
        """Get previously-extracted topics for a collection without re-running extraction."""
        db = get_duckdb_client()
        topic_data = db.get_active_topics(collection_name)
        if not topic_data or not topic_data.get("topics"):
            return None

        topics = _dicts_to_topic_infos(topic_data["topics"])
        num_topics = len([t for t in topics if t.topic_id != -1])
        num_noise = sum(t.count for t in topics if t.topic_id == -1)

        return ExtractTopicsResult(
            collection_name=collection_name,
            num_topics=num_topics,
            num_noise_points=num_noise,
            topics=topics,
            duration_seconds=0.0,
            num_topics_before_reduction=topic_data.get("num_topics_before_reduction"),
            reduction_applied=topic_data.get("reduction_applied", False),
            quality_metrics=topic_data.get("quality_metrics"),
        )

    @strawberry.field
    def collection_probes(self, collection_name: str, info=None) -> CollectionProbesResult:
        """List trained embedding-space probes for a collection."""
        db = get_duckdb_client()
        probes = []
        for p in db.list_probes(collection_name):
            score_field, residual_field = probe_score_field_names(p["target_field"], p["kind"])
            probes.append(
                ProbeInfo(
                    target_field=p["target_field"],
                    kind=p["kind"],
                    score_field=score_field,
                    residual_field=residual_field,
                    metrics=p["metrics"],
                    n_train=p["n_train"] or 0,
                    n_val=p["n_val"] or 0,
                    created_at=p["created_at"],
                    target_mapping=(p["config"] or {}).get("target_mapping"),
                )
            )
        return CollectionProbesResult(collection_name=collection_name, probes=probes)

    @strawberry.field
    async def probe_scores(
        self, collection_name: str, target_field: str, kind: str, info=None
    ) -> ProbeScores | None:
        """Per-item scores for one trained probe (parallel arrays), or None.

        Async: this can read ~100k+ rows into Python lists, which would
        otherwise block the event loop (and stall WebSocket streams).
        """
        db = get_duckdb_client()
        data = await asyncio.to_thread(db.get_probe_scores, collection_name, target_field, kind)
        if data is None:
            return None
        residuals = data["residuals"]
        if all(r is None for r in residuals):
            residuals = None
        return ProbeScores(
            item_ids=data["item_ids"],
            scores=data["scores"],
            residuals=residuals,
        )

    @strawberry.field
    def embeddings(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int = 0,
        filters: list[FilterInput] | None = None,
        include_embeddings: bool = True,
        include_documents: bool = True,
        include_metadata: bool = True,
        info=None,
    ) -> list[EmbeddingItem]:
        """Get embeddings from a collection with optional filtering.

        Args:
            collection_name: Name of the collection
            limit: Maximum number of items to return
            offset: Number of items to skip
            filters: List of filters to apply
            include_embeddings: Whether to include embedding vectors
            include_documents: Whether to include documents
            include_metadata: Whether to include metadata

        Returns:
            List of embedding items
        """
        db = get_duckdb_client()
        dataset_name = _resolve_dataset_name(db, collection_name)
        ds = db.get_dataset(dataset_name)
        if not ds:
            return []

        # Load items from DuckDB (with optional JSON filtering)
        if filters:
            filter_dicts = [
                {"field": f.field, "operator": f.operator.value, "value": f.value} for f in filters
            ]
            rows_data = db.get_filtered_items(
                dataset_name, filter_dicts, limit=limit, offset=offset
            )
        else:
            rows_data = db.get_filtered_items(dataset_name, [], limit=limit, offset=offset)

        if not rows_data:
            return []

        # If embeddings requested, fetch from ChromaDB
        embedding_map = {}
        if include_embeddings:
            client = get_chromadb_client()
            collection = client.get_collection(collection_name, load_embedding_function=False)
            batch_ids = [r["id"] for r in rows_data]
            emb_result = collection.get(ids=batch_ids, include=["embeddings"])
            for eid, evec in zip(emb_result["ids"], emb_result["embeddings"]):
                embedding_map[eid] = evec

        items = []
        for r in rows_data:
            metadata = r.get("metadata", {})
            item = EmbeddingItem(id=r["id"])
            if include_documents:
                item.document = r.get("document")
            if include_metadata:
                item.word = metadata.get("word")
                item.definition = metadata.get("definition")
                item.pos = metadata.get("pos")
                item.metadata = metadata
            if include_embeddings and r["id"] in embedding_map:
                item.embedding = embedding_map[r["id"]]
            items.append(item)

        return items

    @strawberry.field
    def semantic_search(
        self,
        collection_name: str,
        query: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 10,
        similarity_measure: SimilarityMeasure = SimilarityMeasure.COSINE,
        filters: list[FilterInput] | None = None,
        include_embeddings: bool = False,
        query_prompt: str | None = None,
        info=None,
    ) -> list[SemanticSearchResult]:
        """Perform semantic search on a collection.

        ChromaDB handles vector similarity (returns IDs + distances).
        DuckDB enriches results with documents + metadata.

        Args:
            collection_name: Name of the collection
            query: Text query to search for
            query_embedding: Pre-computed query embedding vector
            n_results: Number of results to return
            similarity_measure: Similarity metric to use
            filters: List of filters to apply
            include_embeddings: Whether to include embedding vectors in results
            query_prompt: Prompt to use for query embedding (can be known name or custom string, overrides collection default)

        Returns:
            List of search results with similarities
        """
        client = get_chromadb_client()
        db = get_duckdb_client()
        dataset_name = _resolve_dataset_name(db, collection_name)

        # Pre-filter via DuckDB if filters provided (ChromaDB has no metadata)
        allowed_ids = None
        if filters:
            filter_dicts = [
                {"field": f.field, "operator": f.operator.value, "value": f.value} for f in filters
            ]
            filtered = db.get_filtered_items(dataset_name, filter_dicts, limit=100000)
            allowed_ids = {item["id"] for item in filtered}
            if not allowed_ids:
                return []

        # ChromaDB: vector similarity search (no where clause — metadata not in ChromaDB)
        # Over-fetch when filtering to ensure enough results survive post-filter
        fetch_n = n_results * 5 if allowed_ids else n_results
        results = client.semantic_search(
            collection_name=collection_name,
            query_texts=[query] if query else None,
            query_embeddings=[query_embedding] if query_embedding else None,
            n_results=fetch_n,
            distance_metric=similarity_measure.value,
            query_prompt=query_prompt,
        )

        if not results["ids"]:
            return []

        result_ids = results["ids"][0]

        # DuckDB: enrich with documents + metadata
        items_by_id = {}
        enriched = db.get_items_by_ids(dataset_name, result_ids)
        for item in enriched:
            items_by_id[item["id"]] = item

        # Build results in ChromaDB's ranked order, post-filtering if needed
        search_results = []
        for i, item_id in enumerate(result_ids):
            if allowed_ids is not None and item_id not in allowed_ids:
                continue

            distance = results["distances"][0][i]
            similarity = results["similarities"][0][i]
            enriched_item = items_by_id.get(item_id, {})

            result = SemanticSearchResult(
                id=item_id,
                document=enriched_item.get("document"),
                metadata=enriched_item.get("metadata", {}),
                distance=distance,
                similarity=similarity,
            )

            if include_embeddings and results.get("embeddings"):
                result.embedding = results["embeddings"][0][i]

            search_results.append(result)
            if len(search_results) >= n_results:
                break

        return search_results

    @strawberry.field
    def semantic_search_by_id(
        self,
        collection_name: str,
        item_id: str,
        n_results: int = 10,
        similarity_measure: SimilarityMeasure = SimilarityMeasure.COSINE,
        filters: list[FilterInput] | None = None,
        info=None,
    ) -> list[SemanticSearchResult]:
        """Find similar items using an existing item's embedding.

        ChromaDB: lookup embedding + vector search.
        DuckDB: enrich results with documents + metadata.

        Args:
            collection_name: Name of the collection
            item_id: ID of the item to find similar items for
            n_results: Number of results to return
            similarity_measure: Similarity metric to use
            filters: List of filters to apply

        Returns:
            List of search results with similarities
        """
        client = get_chromadb_client()
        db = get_duckdb_client()
        dataset_name = _resolve_dataset_name(db, collection_name)

        # ChromaDB: get the embedding for the item
        collection = client.get_collection(collection_name, load_embedding_function=False)
        item_data = collection.get(ids=[item_id], include=["embeddings"])

        if item_data is None or "embeddings" not in item_data or len(item_data["embeddings"]) == 0:
            return []

        query_embedding = item_data["embeddings"][0]

        # Pre-filter via DuckDB if filters provided
        allowed_ids = None
        if filters:
            filter_dicts = [
                {"field": f.field, "operator": f.operator.value, "value": f.value} for f in filters
            ]
            filtered = db.get_filtered_items(dataset_name, filter_dicts, limit=100000)
            allowed_ids = {item["id"] for item in filtered}
            if not allowed_ids:
                return []

        # ChromaDB: vector similarity search (no where clause)
        fetch_n = n_results * 5 if allowed_ids else n_results
        results = client.semantic_search(
            collection_name=collection_name,
            query_texts=None,
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            distance_metric=similarity_measure.value,
        )

        if not results["ids"]:
            return []

        result_ids = results["ids"][0]

        # DuckDB: enrich with documents + metadata
        items_by_id = {}
        enriched = db.get_items_by_ids(dataset_name, result_ids)
        for item in enriched:
            items_by_id[item["id"]] = item

        search_results = []
        for i, result_id in enumerate(result_ids):
            if allowed_ids is not None and result_id not in allowed_ids:
                continue
            enriched_item = items_by_id.get(result_id, {})
            search_results.append(
                SemanticSearchResult(
                    id=result_id,
                    document=enriched_item.get("document"),
                    metadata=enriched_item.get("metadata", {}),
                    distance=results["distances"][0][i],
                    similarity=results["similarities"][0][i],
                )
            )
            if len(search_results) >= n_results:
                break

        return search_results

    @strawberry.field
    def text_search(
        self,
        collection_name: str,
        query: str,
        fields: list[str] | None = None,
        mode: TextSearchMode = TextSearchMode.CONTAINS,
        case_sensitive: bool = False,
        filters: list[FilterInput] | None = None,
        limit: int | None = None,
        info=None,
    ) -> TextSearchResponse:
        """Full-text search across document content and/or metadata fields.

        Args:
            collection_name: Name of the collection to search.
            query: The search string.
            fields: Fields to search. Use "__document__" for the embedded text.
                None (default) searches documents only.
            mode: CONTAINS (substring) or EXACT (full value).
            case_sensitive: Whether matching is case-sensitive.
            filters: Optional metadata filters to restrict search scope.
            limit: Maximum matches to return; totalMatches still reports the
                full count so clients can detect overly broad queries.

        Returns:
            TextSearchResponse with matching items.
        """
        db = get_duckdb_client()
        dataset_name = _resolve_dataset_name(db, collection_name)
        filter_dicts = None
        if filters:
            filter_dicts = [
                {"field": f.field, "operator": f.operator.value, "value": f.value} for f in filters
            ]
        result = db.text_search(
            dataset_name=dataset_name,
            query=query,
            fields=fields,
            mode=mode.value,
            case_sensitive=case_sensitive,
            filters=filter_dicts,
            limit=limit,
        )

        matches = [
            TextSearchMatch(
                id=m["id"],
                matched_field=m["matched_field"],
                snippet=m.get("snippet"),
            )
            for m in result["matches"]
        ]

        return TextSearchResponse(
            matches=matches,
            total_matches=result["total_matches"],
        )

    # ------------------------------------------------------------------
    # SAE (Sparse Autoencoder) queries
    # ------------------------------------------------------------------

    @strawberry.field
    def sae_models(self, info) -> list[SaeModelInfo]:
        """List available SAE model/layer combinations with counts."""
        db = get_duckdb_client()
        rows = db.list_sae_models()
        return [
            SaeModelInfo(
                model_id=r["model_id"],
                sae_id=r["sae_id"],
                feature_count=r["feature_count"],
                activation_count=r["activation_count"],
            )
            for r in rows
        ]

    @strawberry.field
    def sae_feature(
        self, model_id: str, sae_id: str, feature_index: int, info=None
    ) -> SaeFeature | None:
        """Get a single SAE feature with metadata and logits."""
        db = get_duckdb_client()
        row = db.get_sae_feature(model_id, sae_id, feature_index)
        if not row:
            return None
        return _dict_to_sae_feature(row)

    @strawberry.field
    def sae_activations(
        self,
        model_id: str,
        sae_id: str,
        feature_index: int,
        limit: int = 20,
        info=None,
    ) -> list[SaeActivation]:
        """Get top activations for a feature, ordered by max value."""
        db = get_duckdb_client()
        rows = db.get_sae_activations(model_id, sae_id, feature_index, limit)
        return [
            SaeActivation(
                id=r["id"],
                tokens=r["tokens"],
                values=r["values"],
                max_value=r["max_value"],
                max_value_token_index=r["max_value_token_index"],
            )
            for r in rows
        ]

    @strawberry.field
    def sae_feature_search(
        self,
        model_id: str | None = None,
        sae_id: str | None = None,
        sae_ids: list[str] | None = None,
        query: str | None = None,
        min_density: float | None = None,
        max_density: float | None = None,
        limit: int = 50,
        offset: int = 0,
        info=None,
    ) -> list[SaeFeatureSearchResult]:
        """Search features by label text and/or density range.

        All filter params are optional for cross-SAE search.
        ``sae_ids`` filters to a set of SAEs (takes precedence over ``sae_id``).
        """
        db = get_duckdb_client()
        features = db.search_sae_features(
            model_id=model_id,
            sae_id=sae_id,
            sae_ids=sae_ids,
            query=query,
            min_density=min_density,
            max_density=max_density,
            limit=limit,
            offset=offset,
        )
        return [SaeFeatureSearchResult(feature=_dict_to_sae_feature(f)) for f in features]

    @strawberry.field
    def sae_feature_densities(
        self,
        model_id: str,
        sae_id: str,
        info=None,
    ) -> list[float]:
        """Return all density values for a model/sae pair (for histogram)."""
        db = get_duckdb_client()
        return db.get_sae_feature_densities(model_id, sae_id)

    @strawberry.field
    def sae_activations_by_quantile(
        self,
        model_id: str,
        sae_id: str,
        feature_index: int,
        n_quantiles: int = 5,
        per_quantile_limit: int = 5,
        info=None,
    ) -> list[SaeActivationQuantileGroup]:
        """Return activations grouped by quantile bins."""
        db = get_duckdb_client()
        groups = db.get_sae_activations_by_quantile(
            model_id,
            sae_id,
            feature_index,
            n_quantiles,
            per_quantile_limit,
        )
        return [
            SaeActivationQuantileGroup(
                quantile=g["quantile"],
                bin_min=g["bin_min"],
                bin_max=g["bin_max"],
                activations=[
                    SaeActivation(
                        id=a["id"],
                        tokens=a["tokens"],
                        values=a["values"],
                        max_value=a["max_value"],
                        max_value_token_index=a["max_value_token_index"],
                    )
                    for a in g["activations"]
                ],
            )
            for g in groups
        ]

    # ------------------------------------------------------------------
    # Interpret / SAE inference
    # ------------------------------------------------------------------

    @strawberry.field
    def model_status(self, info=None) -> ModelStatus:
        """Check if the interpretability model is loaded."""
        service = get_interpret_service()
        status = service.get_status()
        return ModelStatus(
            loaded=status.loaded,
            model_name=status.model_name,
            device=status.device,
            variant=status.variant,
            model_size=status.model_size,
        )

    # ------------------------------------------------------------------
    # SAE document activation search
    # ------------------------------------------------------------------

    @strawberry.field
    def search_documents_by_features(
        self,
        collection_name: str,
        query: str,
        model_id: str | None = None,
        sae_id: str | None = None,
        limit: int = 50,
        info=None,
    ) -> DocumentActivationSearchResponse:
        """Two-hop search: feature label text -> matching features -> ranked documents.

        If model_id/sae_id not provided, looks them up from the collection's metadata.
        """
        db = get_duckdb_client()

        # Resolve model_id / sae_id from collection metadata if not provided
        if model_id is None or sae_id is None:
            ds = db.get_dataset(collection_name)
            if ds is None:
                # Try via vector_collections → dataset_name
                vc_dataset = db.get_dataset_name_for_collection(collection_name)
                if vc_dataset:
                    ds = db.get_dataset(vc_dataset)
            if ds:
                raw_extra = ds.get("extra_metadata") or {}
                extra = json.loads(raw_extra) if isinstance(raw_extra, str) else raw_extra
                model_id = model_id or extra.get("sae_model_id")
                sae_id = sae_id or extra.get("sae_id")

        if not model_id or not sae_id:
            return DocumentActivationSearchResponse(
                results=[],
                total_results=0,
                matched_feature_count=0,
                error="Could not resolve model_id/sae_id for this collection.",
            )

        result = db.search_documents_by_feature_labels(
            collection_name=collection_name,
            query=query,
            model_id=model_id,
            sae_id=sae_id,
            limit=limit,
        )

        return DocumentActivationSearchResponse(
            results=[
                DocumentActivationResult(
                    item_id=r["item_id"],
                    document=r.get("document"),
                    metadata=r.get("metadata"),
                    score=r["score"],
                    matching_features=r["matching_features"],
                    row_index=r.get("row_index"),
                )
                for r in result["results"]
            ],
            total_results=len(result["results"]),
            matched_feature_count=result["matched_feature_count"],
            matched_features=[_dict_to_sae_feature(f) for f in result.get("matched_features", [])]
            or None,
        )

    @strawberry.field
    def search_documents_by_feature_indices(
        self,
        collection_name: str,
        feature_indices: list[int],
        limit: int = 50,
        ranking: DocumentRankingMode = DocumentRankingMode.SCALED_SUM,
        info=None,
    ) -> DocumentActivationSearchResponse:
        """Search documents by explicit SAE feature indices (user-selected from combobox).

        Unlike searchDocumentsByFeatures which does label→features→docs in one call,
        this takes pre-selected feature indices directly. Used when the frontend lets
        the user pick specific features from a multi-select combobox.
        ``totalResults`` is the true number of matching documents (pre-limit).
        """
        db = get_duckdb_client()
        result = db.search_documents_by_feature_indices(
            collection_name=collection_name,
            feature_indices=feature_indices,
            limit=limit,
            ranking=ranking.value,
        )
        return DocumentActivationSearchResponse(
            results=[
                DocumentActivationResult(
                    item_id=r["item_id"],
                    document=r.get("document"),
                    metadata=r.get("metadata"),
                    score=r["score"],
                    matching_features=r["matching_features"],
                    row_index=r.get("row_index"),
                )
                for r in result["results"]
            ],
            total_results=result["total_matches"],
            matched_feature_count=len(feature_indices),
        )

    @strawberry.field
    def has_document_activations(
        self,
        collection_name: str,
        info=None,
    ) -> bool:
        """Check if a collection has precomputed SAE document activations."""
        db = get_duckdb_client()
        return db.has_document_activations(collection_name)

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    @strawberry.field
    def chat_sessions(self, limit: int = 50, info=None) -> list[ChatSessionInfo]:
        """List chat sessions ordered by most recently updated."""
        db = get_duckdb_client()
        sessions = db.list_chat_sessions(limit)
        return [
            ChatSessionInfo(
                id=s["id"],
                title=s["title"],
                config=s["config"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
            )
            for s in sessions
        ]

    @strawberry.field
    def chat_session(self, id: str, info=None) -> ChatSessionDetail | None:
        """Get a chat session with all its messages."""
        db = get_duckdb_client()
        data = db.get_chat_session_with_messages(id)
        if not data:
            return None
        return ChatSessionDetail(
            id=data["id"],
            title=data["title"],
            config=data["config"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=[
                ChatSessionMessage(
                    id=m["id"],
                    session_id=m["session_id"],
                    role=m["role"],
                    content=m["content"],
                    parts=m.get("parts"),
                    created_at=m["created_at"],
                )
                for m in data["messages"]
            ],
        )


def _dict_to_sae_feature(d: dict) -> SaeFeature:
    """Convert a DuckDB feature dict to a Strawberry SaeFeature."""
    top = d.get("top_logits") or []
    bottom = d.get("bottom_logits") or []
    return SaeFeature(
        model_id=d["model_id"],
        sae_id=d["sae_id"],
        feature_index=d["feature_index"],
        density=d.get("density"),
        label=d.get("label"),
        top_logits=[SaeLogitEntry(token=e["token"], score=e["score"]) for e in top]
        if top
        else None,
        bottom_logits=[SaeLogitEntry(token=e["token"], score=e["score"]) for e in bottom]
        if bottom
        else None,
    )
