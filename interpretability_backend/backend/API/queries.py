"""GraphQL query resolvers for embedding visualization backend."""

import strawberry
from typing import List, Optional

from .types import (
    Collection,
    CollectionMetadata,
    EmbeddingItem,
    EmbeddingJob,
    FilterInput,
    HFConfigInfo,
    HFDatasetInfo,
    HFDatasetPreview,
    HFFeatureInfo,
    HFSplitInfo,
    LocalFileInfo,
    LocalFilePreview,
    ProjectionData,
    SemanticSearchResult,
    SimilarityMeasure,
    build_where_clause,
)

# Import clients at module level
from ..clients.huggingface_client import (
    get_dataset_info as hf_get_info,
    get_dataset_preview as hf_get_preview,
)
from ..clients.local_data_client import (
    get_local_file_info as get_local_info,
    get_local_file_preview as get_local_preview,
)
from .chromadb_instance import get_chromadb_client
from ..services.job_state import get_job_state_service, JobStatus


@strawberry.type
class Query:
    """GraphQL query root."""

    @strawberry.field
    def collections(self, info) -> List[Collection]:
        """List all available collections.

        Returns:
            List of collections with metadata
        """
        client = get_chromadb_client()
        collections = client.list_collections()

        return [
            Collection(
                name=col["name"],
                metadata=col["metadata"],
                count=col["count"]
            )
            for col in collections
        ]

    @strawberry.field
    def embedding_jobs(self, status: Optional[str] = None, info=None) -> List[EmbeddingJob]:
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
                embedding_model=job.config.get("embedding_model", {}).get("model_name")
                    if job.config.get("embedding_model") else None,
                batch_size=job.config.get("batch_size", 100),
                started_at=job.started_at,
                # Full config for resume verification
                config=job.config
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
            splits = [HFSplitInfo(name=s.name, num_rows=s.num_rows, num_bytes=s.num_bytes)
                      for s in cfg.splits]
            features = [HFFeatureInfo(name=f.name, dtype=f.dtype, description=f.description)
                        for f in cfg.features]
            configs.append(HFConfigInfo(name=cfg.name, splits=splits, features=features))

        return HFDatasetInfo(
            dataset_id=result.dataset_id,
            description=result.description,
            license=result.license,
            configs=configs,
            default_config=result.default_config,
            error=result.error
        )

    @strawberry.field
    def huggingface_dataset_preview(
        self,
        dataset_id: str,
        config: Optional[str] = None,
        split: str = "train",
        n_rows: int = 5,
        info=None
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
            error=result.error
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
            error=result.error
        )

    @strawberry.field
    def local_file_preview(
        self,
        file_path: str,
        n_rows: int = 5,
        info=None
    ) -> LocalFilePreview:
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
            error=result.error
        )

    @strawberry.field
    def collection(self, name: str, info) -> Optional[ProjectionData]:
        """Get complete projection data for a collection.

        Args:
            name: Collection name

        Returns:
            Projection data with PCA/UMAP projections from ChromaDB metadata.
            Generic structure - no hardcoded field names.
        """
        client = get_chromadb_client()

        # Load projection data directly from ChromaDB metadata
        projection_data = client.get_projection_data(name)

        return ProjectionData(
            ids=projection_data["ids"],
            documents=projection_data["documents"],
            item_metadata=projection_data["item_metadata"],
            available_fields=projection_data["available_fields"],
            pca_2d=projection_data["pca_2d"],
            pca_3d=projection_data["pca_3d"],
            umap_2d=projection_data["umap_2d"],
            umap_3d=projection_data["umap_3d"],
            metadata=CollectionMetadata(**projection_data["metadata"])
        )

    @strawberry.field
    def embeddings(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[List[FilterInput]] = None,
        include_embeddings: bool = True,
        include_documents: bool = True,
        include_metadata: bool = True,
        info=None
    ) -> List[EmbeddingItem]:
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
        client = get_chromadb_client()

        # Build include list
        include = []
        if include_embeddings:
            include.append("embeddings")
        if include_documents:
            include.append("documents")
        if include_metadata:
            include.append("metadatas")

        # Build where clause
        where = build_where_clause(filters)

        # Get results
        results = client.get_all_items(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            where=where,
            include=include
        )

        # Convert to EmbeddingItem list
        items = []
        for i, item_id in enumerate(results["ids"]):
            item = EmbeddingItem(id=item_id)

            if "embeddings" in results and results["embeddings"]:
                item.embedding = results["embeddings"][i]

            if "documents" in results and results["documents"]:
                item.document = results["documents"][i]

            if "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][i]
                item.word = metadata.get("word")
                item.definition = metadata.get("definition")
                item.pos = metadata.get("pos")
                item.metadata = metadata

            items.append(item)

        return items

    @strawberry.field
    def semantic_search(
        self,
        collection_name: str,
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 10,
        similarity_measure: SimilarityMeasure = SimilarityMeasure.COSINE,
        filters: Optional[List[FilterInput]] = None,
        include_embeddings: bool = False,
        query_prompt: Optional[str] = None,
        query_prompt_name: Optional[str] = None,
        info=None
    ) -> List[SemanticSearchResult]:
        """Perform semantic search on a collection.

        Args:
            collection_name: Name of the collection
            query: Text query to search for
            query_embedding: Pre-computed query embedding vector
            n_results: Number of results to return
            similarity_measure: Similarity metric to use
            filters: List of filters to apply
            include_embeddings: Whether to include embedding vectors in results
            query_prompt: Direct prompt string to use for query embedding (overrides collection default)
            query_prompt_name: Predefined prompt name for query embedding (e.g., "Retrieval-query")

        Returns:
            List of search results with similarities
        """
        client = get_chromadb_client()

        # Build where clause
        where = build_where_clause(filters)

        # Perform search
        results = client.semantic_search(
            collection_name=collection_name,
            query_texts=[query] if query else None,
            query_embeddings=[query_embedding] if query_embedding else None,
            n_results=n_results,
            where=where,
            distance_metric=similarity_measure.value,
            query_prompt=query_prompt,
            query_prompt_name=query_prompt_name
        )

        # Convert to SemanticSearchResult list
        search_results = []
        if results["ids"]:
            for i, item_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                document = results["documents"][0][i] if results.get("documents") else None
                distance = results["distances"][0][i]
                similarity = results["similarities"][0][i]

                result = SemanticSearchResult(
                    id=item_id,
                    document=document,
                    metadata=metadata,
                    distance=distance,
                    similarity=similarity
                )

                if include_embeddings and results.get("embeddings"):
                    result.embedding = results["embeddings"][0][i]

                search_results.append(result)

        return search_results

    @strawberry.field
    def semantic_search_by_id(
        self,
        collection_name: str,
        item_id: str,
        n_results: int = 10,
        similarity_measure: SimilarityMeasure = SimilarityMeasure.COSINE,
        filters: Optional[List[FilterInput]] = None,
        info=None
    ) -> List[SemanticSearchResult]:
        """Find similar items using an existing item's embedding.

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
        # Don't load EF - we're using pre-computed embeddings
        collection = client.get_collection(collection_name, load_embedding_function=False)

        # Get the embedding for the item
        item_data = collection.get(
            ids=[item_id],
            include=["embeddings"]
        )

        # Check if embeddings exist and have data
        if (item_data is None or
            "embeddings" not in item_data or
            len(item_data["embeddings"]) == 0):
            return []

        query_embedding = item_data["embeddings"][0]

        # Build where clause
        where = build_where_clause(filters)

        # Perform search using the item's embedding
        results = client.semantic_search(
            collection_name=collection_name,
            query_texts=None,
            query_embeddings=[query_embedding],
            n_results=n_results + 1,  # +1 to account for the item itself
            where=where,
            distance_metric=similarity_measure.value
        )

        # Convert to SemanticSearchResult list, excluding the query item itself
        search_results = []
        if results["ids"]:
            for i, result_id in enumerate(results["ids"][0]):
                # Skip the item itself
                if result_id == item_id:
                    continue

                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                document = results["documents"][0][i] if results.get("documents") else None
                distance = results["distances"][0][i]
                similarity = results["similarities"][0][i]

                result = SemanticSearchResult(
                    id=result_id,
                    document=document,
                    metadata=metadata,
                    distance=distance,
                    similarity=similarity
                )

                search_results.append(result)

                # Stop once we have enough results (excluding the query item)
                if len(search_results) >= n_results:
                    break

        return search_results
