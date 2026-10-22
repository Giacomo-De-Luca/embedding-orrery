"""Multimodal embedding: images, audio, video and rows mixing them with text.

``MultimodalEmbeddingRun`` embeds the ``DataType.MULTIMODAL`` jobs of both the
local-file and the HuggingFace pipelines (``embed_local_file`` /
``embed_huggingface`` dispatch to it with a ``MediaSource`` describing their
rows). One run:

1. detects the kind of each media column (``MediaColumn.detect``: dataset dtype
   hints, magic bytes, extensions);
2. rejects options whose worst-case row exceeds the model's context
   (``MediaEmbeddingSettings.check_budget``), before any model loads;
3. loads the model with only the encoder towers those kinds need (an audio
   dataset loads no vision tower) and checks it embeds them
   (``require_modalities``);
4. per batch, decodes each row's media (``MediaDecoder``), builds one input
   with its text (``MediaRowBuilder``: the task prefix on the text only) and
   embeds the batch (``embed_inputs``: a sentence-transformers prompt applies
   to rows with text only);
5. stores the vectors in ChromaDB and the documents in DuckDB, with each
   media cell replaced by its path or a placeholder in the item metadata.

Rows whose media cannot be decoded are skipped and reported as a warning;
rows keep their source order, so ids stay stable when a job is resumed.
Job state, progress, cancellation and the result come from ``embedding_job``.
"""

import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..services.progress_emitter import emit_progress_sync
from ..utils.color_preprocessing import preprocess_color_metadata
from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items
from ..utils.embedding_model_catalog import EmbeddingModelCatalog
from ..utils.embedding_tasks import EmbeddingTexts
from ..utils.id_utils import IDDeduplicator
from ..utils.media_decoding import MediaDecodeError, MediaDecoder
from ..utils.media_kinds import MediaKindDetector
from ..utils.media_rows import MediaColumn, MediaRowBuilder
from ..utils.media_settings import MediaEmbeddingSettings
from ..utils.text_processing import extract_metadata
from .config import DataType, EmbeddingModelConfig, EmbeddingResult, MediaSourceConfig
from .create_embedding_function import create_embedding_function
from .embedding_job import EmbeddingBatchJob, open_embedding_collection

logger = logging.getLogger("orrery")

# Skipped rows named in the log; the rest are only counted.
_LOGGED_FAILURES = 5


@dataclass(frozen=True)
class MediaSource:
    """What a pipeline tells the multimodal run about its rows."""

    job_type: str  # "local_file" or "huggingface"
    # Source keys of the collection metadata (description, source file or dataset, ...).
    collection_metadata: Mapping[str, Any]
    # The id of a row without an id column value.
    auto_id: Callable[[Mapping[str, Any], int], str]
    # Where relative media paths resolve (a local file's folder).
    base_dir: Path | None = None
    # Dataset dtype labels per column (HF "image" / "audio" / "video").
    column_hints: Mapping[str, str] = field(default_factory=dict)
    # Internal row keys never stored as metadata (the HF split tag).
    excluded_columns: frozenset[str] = frozenset()
    # Extra metadata per row (the HF source split).
    row_metadata: Callable[[Mapping[str, Any]], dict] | None = None


def default_multimodal_model() -> EmbeddingModelConfig:
    """The catalog's multimodal default (EmbeddingGemma 2), as a model config."""
    model = EmbeddingModelCatalog.shipped().default_model_for(DataType.MULTIMODAL.value)
    return EmbeddingModelConfig(model_name=model.model)


class MultimodalEmbeddingRun:
    """One multimodal embedding job (see module docstring)."""

    def __init__(
        self,
        *,
        client,
        config: MediaSourceConfig,
        rows: list[dict],
        total: int,
        device: str,
        start_time: float,
        source: MediaSource,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
        settings: MediaEmbeddingSettings | None = None,
        embedding_function_factory: Callable[..., tuple[Any, int]] = create_embedding_function,
    ):
        self.client = client
        self.config = config
        self.rows = rows
        self.total = total
        self.device = device
        self.start_time = start_time
        self.source = source
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event
        self.settings = settings or MediaEmbeddingSettings.shipped()
        self.factory = embedding_function_factory
        self.model_config = config.embedding_model or default_multimodal_model()
        self.options = config.media_options or self.settings.defaults
        self.skipped = 0
        self.first_failure: str | None = None

    def run(self) -> EmbeddingResult:
        config, model_config = self.config, self.model_config
        columns = MediaColumn.detect(
            self.rows,
            config.media_columns or [],
            hints=self.source.column_hints,
            detector=MediaKindDetector(self.settings.extensions()),
            sample_size=self.settings.detection_sample_rows,
        )
        self.settings.check_budget(self.options, Counter(column.kind for column in columns))
        text_columns = list(config.columns or [])
        texts = None
        if text_columns:
            texts = EmbeddingTexts(
                text_columns, config.text_template, model_config.prompt_format, model_config.prompt
            )
            texts.validate(self.rows)
        builder = MediaRowBuilder(
            columns,
            MediaDecoder(self.settings, self.options, base_dir=self.source.base_dir),
            texts,
        )
        kinds = builder.kinds
        print(f"Embedding media columns {[(c.name, c.kind.value) for c in columns]}")

        self._status(f"Loading embedding model ({model_config.model_name})...")
        embedding_func, embedding_dim = self.factory(
            model_config, self.device, modalities=("text", *(kind.value for kind in kinds))
        )
        embedding_func.require_modalities([kind.value for kind in kinds], with_text=bool(texts))

        collection_metadata = {
            **self.source.collection_metadata,
            "embedded_columns": json.dumps(text_columns),
            "media_columns": json.dumps({c.name: c.kind.value for c in columns}),
            "media_options": json.dumps(self.options.to_dict()),
            "data_type": DataType.MULTIMODAL.value,
            **model_config.collection_metadata(embedding_dim),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        collection, existing_ids = open_embedding_collection(
            self.client,
            config.collection_name,
            resume=config.resume,
            metadata=collection_metadata,
            embedding_function=embedding_func,
        )
        dataset_name = sync_dataset_and_collection(
            config.collection_name,
            collection_metadata,
            model_config=model_config,
            embedding_dim=embedding_dim,
        )
        metadata_columns = self._metadata_columns(text_columns)

        job = EmbeddingBatchJob(
            collection_name=config.collection_name,
            job_type=self.source.job_type,
            job_config=config.to_job_config(),
            n_rows=len(self.rows),
            batch_size=min(config.batch_size, self.settings.rows_per_batch),
            embedding_dim=embedding_dim,
            device=self.device,
            start_time=self.start_time,
            model_config=model_config,
            cancel_event=self.cancel_event,
            progress_callback=self.progress_callback,
            description="Embedding media batches",
        )
        job.start(already_embedded=len(existing_ids), resume=config.resume)
        processing_kwargs = self.options.processing_kwargs(kinds)
        ids_seen = IDDeduplicator()

        for batch_start, batch_end in job.batches():
            ids, documents, inputs, has_text, metadatas = [], [], [], [], []
            for row_idx in range(batch_start, batch_end):
                row = self.rows[row_idx]
                doc_id = self._row_id(row, row_idx, ids_seen)
                if doc_id in existing_ids:
                    continue
                try:
                    media_row = builder.build(row)
                except MediaDecodeError as error:
                    self._skip(doc_id, error)
                    continue
                if media_row is None:
                    continue
                metadata = extract_metadata(row, metadata_columns)
                metadata.update(media_row.references)
                if self.source.row_metadata is not None:
                    metadata.update(self.source.row_metadata(row))
                metadata = preprocess_color_metadata(metadata, row)
                metadata["row_index"] = row_idx

                ids.append(doc_id)
                documents.append(media_row.document)
                inputs.append(media_row.embed_input)
                has_text.append(media_row.has_text)
                metadatas.append(metadata)

            if ids:
                vectors = embedding_func.embed_inputs(
                    inputs,
                    has_text,
                    processing_kwargs=processing_kwargs,
                    batch_size=self.settings.encode_batch_size,
                )
                collection.add(ids=ids, embeddings=vectors)
                if dataset_name:
                    sync_items(dataset_name, ids, documents, metadatas)
            job.record(added=len(ids), end=batch_end)

        result = job.result()
        if self.skipped:
            result.warnings.append(
                f"{self.skipped} row(s) skipped because their media could not be decoded "
                f"(first: {self.first_failure})"
            )
        return result

    def _metadata_columns(self, text_columns: list[str]) -> list[str]:
        """The configured metadata columns, else every column but the text and id ones."""
        if self.config.metadata_columns is not None:
            return list(self.config.metadata_columns)
        excluded = set(text_columns) | set(self.source.excluded_columns)
        if self.config.id_column:
            excluded.add(self.config.id_column)
        keys = dict.fromkeys(key for row in self.rows[:1] for key in row)
        return [key for key in keys if key not in excluded]

    def _row_id(self, row: Mapping[str, Any], row_idx: int, ids_seen: IDDeduplicator) -> str:
        id_column = self.config.id_column
        if id_column and row.get(id_column) is not None:
            return ids_seen.get_unique_id(str(row[id_column]))
        return self.source.auto_id(row, row_idx)

    def _skip(self, doc_id: str, error: MediaDecodeError) -> None:
        self.skipped += 1
        if self.first_failure is None:
            self.first_failure = f"{doc_id}: {error}"
        if self.skipped <= _LOGGED_FAILURES:
            logger.warning("Skipping %s: %s", doc_id, error)

    def _status(self, message: str) -> None:
        emit_progress_sync(
            job_id=self.config.collection_name,
            status="running",
            items_processed=0,
            total_items=len(self.rows),
            current_batch=0,
            total_batches=0,
            message=message,
        )
