"""Ingestion pipeline for SAE (Sparse Autoencoder) feature and activation data.

Two entry points:
- ``ingest_sae_features`` — reads a parquet file of SAE feature metadata
  (index, density, label, top/bottom logits, explanation-embedding vector)
  into DuckDB ``sae_features`` and optionally stores the explanation vectors
  in ChromaDB (+ DuckDB items table) for semantic search / visualization.
- ``ingest_sae_activations`` — streams a JSONL file of per-feature activation
  examples into DuckDB ``sae_activations``.

Both share the composite key ``(model_id, sae_id, feature_index)`` for joins.
"""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("star_map." + __name__)

ACTIVATION_BATCH_SIZE = 5_000
VECTOR_BATCH_SIZE = 500


# ── helpers ──────────────────────────────────────────────────────────


def _serialize_logits(value) -> str:
    """Convert a logit column value (numpy array of dicts, list, or str) to JSON."""
    if isinstance(value, str):
        return value
    if value is None:
        return "[]"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps([{"token": d["token"], "score": float(d["score"])} for d in value])


def _parse_vector(value) -> list[float] | None:
    """Convert a vector cell (numpy array, list, or JSON string) to List[float]."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ── feature ingestion ────────────────────────────────────────────────


def ingest_sae_features(
    parquet_path: str,
    model_id: str,
    sae_id: str,
    store_vectors: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Load SAE feature parquet into DuckDB (+ optional ChromaDB vectors).

    Args:
        parquet_path: Path to the features parquet
            (columns: index, vector, density, label, top_logits, bottom_logits).
        model_id: Model identifier, e.g. ``"gemma-3-4b-it"``.
        sae_id: SAE identifier, e.g. ``"9-gemmascope-2-res-16k"``.
        store_vectors: If True, store explanation-embedding vectors in ChromaDB
            and register a dataset + vector_collection for visualization.
        progress_callback: ``(done, total)`` progress reporter.

    Returns:
        Dict with keys: model_id, sae_id, features_inserted,
        vectors_stored, duration_seconds, error.
    """
    start = time.time()
    result: dict[str, Any] = {
        "model_id": model_id,
        "sae_id": sae_id,
        "records_inserted": 0,
        "duration_seconds": 0.0,
        "error": None,
    }

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        result["error"] = f"Failed to read parquet: {e}"
        return result

    required = {"index", "density", "label", "top_logits", "bottom_logits"}
    missing = required - set(df.columns)
    if missing:
        result["error"] = f"Missing columns in parquet: {missing}"
        return result

    total = len(df)
    logger.info("Ingesting %d SAE features from %s", total, parquet_path)

    # ---- 1. Serialize logit columns to JSON strings ----
    feature_df = pd.DataFrame(
        {
            "feature_index": df["index"].astype(int),
            "density": df["density"].astype(float),
            "label": df["label"].astype(str),
            "top_logits": df["top_logits"].apply(_serialize_logits),
            "bottom_logits": df["bottom_logits"].apply(_serialize_logits),
        }
    )

    # ---- 2. Insert into sae_features ----
    from ..API.duckdb_instance import get_duckdb_client

    db = get_duckdb_client()
    count = db.insert_sae_features_batch(model_id, sae_id, feature_df)
    result["records_inserted"] = count

    if progress_callback:
        progress_callback(total // 2, total)  # halfway after features

    # ---- 3. Optionally store explanation vectors in ChromaDB ----
    if store_vectors and "vector" in df.columns:
        collection_name = f"sae_{model_id}_{sae_id}".replace("/", "_")
        _store_feature_vectors(
            db,
            df,
            model_id,
            sae_id,
            collection_name,
            parquet_path,
            progress_callback,
            total,
        )

    if progress_callback:
        progress_callback(total, total)

    result["duration_seconds"] = round(time.time() - start, 2)
    logger.info(
        "SAE feature ingestion done in %.1fs — %d features", result["duration_seconds"], count
    )
    return result


def _store_feature_vectors(
    db,
    df: pd.DataFrame,
    model_id: str,
    sae_id: str,
    collection_name: str,
    parquet_path: str,
    progress_callback,
    total: int,
):
    """Store explanation-embedding vectors in ChromaDB and DuckDB items table."""
    from ..API.chromadb_instance import get_chromadb_client
    from ..utils.duckdb_sync import sync_dataset_and_collection, sync_items

    chroma = get_chromadb_client()

    # Detect dimension from first row
    first_vec = _parse_vector(df["vector"].iloc[0])
    if first_vec is None:
        logger.warning("Could not parse explanation vectors — skipping ChromaDB storage")
        return
    embedding_dim = len(first_vec)

    collection_metadata = {
        "description": f"SAE feature explanation embeddings ({model_id}/{sae_id})",
        "source_file": str(parquet_path),
        "data_type": "vector",
        "embedding_provider": "neuronpedia",
        "embedding_model": f"sae_explanations_{sae_id}",
        "embedding_dim": embedding_dim,
        "is_sae_collection": True,
        "sae_model_id": model_id,
        "sae_id": sae_id,
    }

    # Create ChromaDB collection (no embedding function — vectors are pre-computed)
    try:
        chroma.client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = chroma.client.create_collection(
        name=collection_name,
        metadata=collection_metadata,
    )

    # Register in DuckDB datasets + vector_collections
    sync_dataset_and_collection(
        collection_name,
        collection_metadata,
        embedding_dim=embedding_dim,
    )

    # Batch-insert vectors into ChromaDB and items into DuckDB
    for batch_start in tqdm(
        range(0, len(df), VECTOR_BATCH_SIZE),
        desc="Storing SAE vectors",
        unit="batch",
    ):
        batch = df.iloc[batch_start : batch_start + VECTOR_BATCH_SIZE]

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for _, row in batch.iterrows():
            vec = _parse_vector(row["vector"])
            if vec is None:
                continue
            ids.append(str(int(row["index"])))
            embeddings.append(vec)
            documents.append(str(row["label"]) if pd.notna(row["label"]) else "")
            metadatas.append(
                {
                    "feature_index": int(row["index"]),
                    "density": float(row["density"]),
                    "row_index": int(row["index"]),
                }
            )

        if ids:
            collection.add(ids=ids, embeddings=embeddings)
            sync_items(collection_name, ids, documents, metadatas)

        if progress_callback:
            done = total // 2 + (batch_start + len(batch)) * total // (2 * len(df))
            progress_callback(done, total)


# ── activation ingestion ─────────────────────────────────────────────


def ingest_sae_activations(
    jsonl_path: str,
    model_id: str | None = None,
    sae_id: str | None = None,
    batch_size: int = ACTIVATION_BATCH_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Stream SAE activation JSONL into DuckDB.

    Args:
        jsonl_path: Path to the activations JSONL.
        model_id: Override model ID (auto-detected from ``modelId`` field if None).
        sae_id: Override SAE ID (auto-detected from ``layer`` field if None).
        batch_size: Records per batch insert.
        progress_callback: ``(done, total)`` progress reporter.

    Returns:
        Dict with keys: model_id, sae_id, records_inserted,
        duration_seconds, error.
    """
    start = time.time()
    result: dict[str, Any] = {
        "model_id": model_id or "",
        "sae_id": sae_id or "",
        "records_inserted": 0,
        "duration_seconds": 0.0,
        "error": None,
    }

    path = Path(jsonl_path)
    if not path.exists():
        result["error"] = f"File not found: {jsonl_path}"
        return result

    # Count lines for progress (fast scan)
    with open(path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    logger.info("Ingesting %d activation records from %s", total_lines, jsonl_path)

    from ..API.duckdb_instance import get_duckdb_client

    db = get_duckdb_client()

    inserted = 0
    skipped = 0
    batch_rows: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(
            tqdm(f, total=total_lines, desc="SAE activations", unit="rec")
        ):
            try:
                entry = json.loads(line)

                row_model_id = model_id or entry.get("modelId", "")
                row_sae_id = sae_id or entry.get("layer", "")

                # Update result metadata from first record
                if line_no == 0:
                    result["model_id"] = row_model_id
                    result["sae_id"] = row_sae_id

                batch_rows.append(
                    {
                        "id": entry["id"],
                        "model_id": row_model_id,
                        "sae_id": row_sae_id,
                        "feature_index": int(entry["index"]),
                        "tokens": json.dumps(entry["tokens"]),
                        "act_values": json.dumps(entry.get("values", [])),
                        "max_value": float(entry.get("maxValue", 0)),
                        "max_value_token_idx": int(entry.get("maxValueTokenIndex", 0)),
                        "min_value": float(entry.get("minValue", 0)),
                        "qualifying_token_idx": int(entry.get("qualifyingTokenIndex", 0)),
                    }
                )
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                skipped += 1
                if skipped <= 5:
                    logger.warning("Skipping malformed activation line %d: %s", line_no, e)
                continue

            if len(batch_rows) >= batch_size:
                batch_df = pd.DataFrame(batch_rows)
                db.insert_sae_activations_batch(batch_df)
                inserted += len(batch_rows)
                batch_rows = []

                if progress_callback:
                    progress_callback(inserted, total_lines)

    # Flush remaining
    if batch_rows:
        batch_df = pd.DataFrame(batch_rows)
        db.insert_sae_activations_batch(batch_df)
        inserted += len(batch_rows)

    if progress_callback:
        progress_callback(inserted, total_lines)

    result["records_inserted"] = inserted
    result["duration_seconds"] = round(time.time() - start, 2)
    logger.info(
        "SAE activation ingestion done in %.1fs — %d records", result["duration_seconds"], inserted
    )
    return result
