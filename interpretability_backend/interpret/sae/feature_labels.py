"""Neuronpedia autointerpreter label lookup for SAE features, backed by SQLite.

All label data lives in a single ``features.db`` file in the labels directory.
Each source (model/layer/hook/width) is imported from its JSONL file on first
query, and re-imported automatically if the JSONL file has been modified since
the last import.

Supports multiple labelling methods per feature — the Neuronpedia autointerpreter
label is stored as method="label". Custom methods can be added via write_labels().

Stores 256-dim explanation embeddings for feature-to-feature similarity search,
and top/bottom logits for understanding what tokens each feature promotes/suppresses.
"""

import json
import re
import sqlite3
import struct
from pathlib import Path

import torch

from interpret.sae.sae_config import SAEConfig

# Mapping from HookType values to Neuronpedia filename abbreviations
HOOK_TO_NEURONPEDIA: dict[str, str] = {
    "resid_post": "res",
    "mlp_out": "mlp",
    "attn_out": "att",
}

# Internal key type: (model_id, layer, hook_abbrev, width_str)
type _Key = tuple[str, int, str, str]

_EMBEDDING_DIM = 256

# String-to-integer width mapping (matches loading.WIDTH_TO_D_SAE)
_WIDTH_STR_TO_INT: dict[str, int] = {"16k": 16384, "65k": 65536, "262k": 262144}


def _width_as_int(width: int | str) -> int:
    """Convert a width value to integer form."""
    if isinstance(width, int):
        return width
    if width in _WIDTH_STR_TO_INT:
        return _WIDTH_STR_TO_INT[width]
    raise ValueError(f"Unknown width string: {width!r}")

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    source   TEXT NOT NULL,
    idx      INTEGER NOT NULL,
    density  REAL,
    PRIMARY KEY (source, idx)
);
CREATE TABLE IF NOT EXISTS labels (
    source   TEXT NOT NULL,
    idx      INTEGER NOT NULL,
    method   TEXT NOT NULL,
    text     TEXT NOT NULL,
    PRIMARY KEY (source, idx, method)
);
CREATE TABLE IF NOT EXISTS embeddings (
    source    TEXT NOT NULL,
    idx       INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (source, idx)
);
CREATE TABLE IF NOT EXISTS logits (
    source    TEXT NOT NULL,
    idx       INTEGER NOT NULL,
    direction TEXT NOT NULL,
    token     TEXT NOT NULL,
    score     REAL NOT NULL,
    PRIMARY KEY (source, idx, direction, token)
);
CREATE TABLE IF NOT EXISTS imports (
    source       TEXT PRIMARY KEY,
    jsonl_path   TEXT NOT NULL,
    mtime        REAL NOT NULL,
    num_features INTEGER NOT NULL
);
"""


def _source_id(key: _Key) -> str:
    """Build a source identifier string from a key tuple."""
    model_id, layer, hook, width = key
    return f"{model_id}_{layer}-gemmascope-2-{hook}-{width}"


def _pack_embedding(values: list[float]) -> bytes:
    """Pack a list of floats into a binary blob."""
    return struct.pack(f"{len(values)}f", *values)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob into a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class FeatureLabelStore:
    """SQLite-backed lookup for SAE feature labels and metadata.

    All sources share a single ``features.db`` in the labels directory.
    Each source is imported from its JSONL file on first query, and
    re-imported if the JSONL has been modified since the last import.

    Usage::

        store = FeatureLabelStore("resources/sae_labels/neuronpedia_gemma-3-4b-it")
        config = SAEConfig(layer_index=29)
        model_id, layer, hook, width = store.params_from_config(config)

        label = store.get_label(4287, model_id, layer, hook, width)
        similar = store.find_similar_features(4287, model_id, layer, hook, width)
    """

    def __init__(self, labels_dir: str | Path) -> None:
        self._dir = Path(labels_dir)
        if not self._dir.is_dir():
            raise FileNotFoundError(f"Labels directory not found: {self._dir}")
        self._available = self._scan_files()
        self._db_path = self._dir / "features.db"
        self._conn: sqlite3.Connection | None = None
        self._density_cache: dict[_Key, torch.Tensor] = {}
        self._embedding_cache: dict[_Key, tuple[torch.Tensor, list[int]]] = {}

    def _get_conn(self) -> sqlite3.Connection:
        """Return the shared database connection, creating it if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.executescript(_CREATE_SCHEMA)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "FeatureLabelStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- File discovery ---

    def _scan_files(self) -> dict[_Key, Path]:
        """Scan for JSONL files and index by (model_id, layer, hook, width)."""
        available: dict[_Key, Path] = {}
        pattern = re.compile(
            r"^(.+)_(\d+)-gemmascope-2-(\w+)-(\w+)_features\.jsonl$"
        )
        for f in self._dir.glob("*_features.jsonl"):
            m = pattern.match(f.name)
            if m:
                model_id = m.group(1)
                layer = int(m.group(2))
                hook = m.group(3)
                width = m.group(4)
                available[(model_id, layer, hook, width)] = f
        return available

    def _resolve_key(
        self, model_id: str, layer: int, hook: str, width: int | str,
    ) -> _Key:
        """Convert user-facing params to internal key.

        Width accepts both integer (e.g. 16384) and string (e.g. "16k") forms.
        """
        hook_abbrev = HOOK_TO_NEURONPEDIA.get(hook, hook)
        if isinstance(width, str):
            width_str = width
        else:
            width_str = f"{width // 1024}k" if width >= 1024 else str(width)
        return (model_id, layer, hook_abbrev, width_str)

    # --- Source import management ---

    def _ensure_source(self, model_id: str, layer: int, hook: str, width: int | str) -> str:
        """Ensure a source is imported into the DB, returning its source_id."""
        key = self._resolve_key(model_id, layer, hook, width)
        source = _source_id(key)
        conn = self._get_conn()

        jsonl_path = self._available.get(key)
        if jsonl_path is None:
            row = conn.execute(
                "SELECT 1 FROM imports WHERE source = ?", (source,)
            ).fetchone()
            if row:
                return source
            raise FileNotFoundError(
                f"No label file for model_id={key[0]}, layer={key[1]}, "
                f"hook={key[2]}, width={key[3]}. "
                f"Available: {list(self._available.keys())}"
            )

        current_mtime = jsonl_path.stat().st_mtime

        row = conn.execute(
            "SELECT mtime FROM imports WHERE source = ?", (source,)
        ).fetchone()
        if row and row[0] >= current_mtime:
            return source

        self._import_source(source, jsonl_path, current_mtime)
        self._density_cache.pop(key, None)
        self._embedding_cache.pop(key, None)
        return source

    def _import_source(self, source: str, jsonl_path: Path, mtime: float) -> None:
        """Import a JSONL file into the database for a given source.

        Runs inside an exclusive transaction so readers never see partial data
        and a crash mid-import leaves the DB unchanged.
        """
        conn = self._get_conn()
        conn.execute("BEGIN EXCLUSIVE")
        for table in ("features", "labels", "embeddings", "logits"):
            conn.execute(f"DELETE FROM {table} WHERE source = ?", (source,))

        batch_features: list[tuple[str, int, float]] = []
        batch_labels: list[tuple[str, int, str, str]] = []
        batch_embeddings: list[tuple[str, int, bytes]] = []
        batch_logits: list[tuple[str, int, str, str, float]] = []
        batch_size = 1000
        count = 0
        seen_indices: set[int] = set()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                idx = int(entry["index"])

                # Skip duplicate indices (keep first occurrence)
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)

                density = entry.get("density", 0.0)
                batch_features.append((source, idx, density))

                # Labels
                explanations = entry.get("explanations", [])
                if explanations:
                    label_text = explanations[0].get("text", "")
                    if label_text:
                        batch_labels.append((source, idx, "label", label_text))

                    # Embeddings
                    embedding = explanations[0].get("embedding")
                    if embedding and len(embedding) == _EMBEDDING_DIM:
                        batch_embeddings.append(
                            (source, idx, _pack_embedding(embedding))
                        )

                # Logits
                for direction, key_name in [("top", "top_logits"), ("bottom", "bottom_logits")]:
                    logit_list = entry.get(key_name)
                    if logit_list:
                        for token, score in logit_list:
                            batch_logits.append((source, idx, direction, token, score))

                count += 1
                if count % batch_size == 0:
                    self._flush_batches(
                        conn, batch_features, batch_labels,
                        batch_embeddings, batch_logits,
                    )

        self._flush_batches(
            conn, batch_features, batch_labels,
            batch_embeddings, batch_logits,
        )

        conn.execute(
            "INSERT OR REPLACE INTO imports (source, jsonl_path, mtime, num_features) VALUES (?, ?, ?, ?)",
            (source, str(jsonl_path), mtime, count),
        )
        conn.commit()

    @staticmethod
    def _flush_batches(
        conn: sqlite3.Connection,
        features: list, labels: list,
        embeddings: list, logits: list,
    ) -> None:
        """Flush all batch lists to the database and clear them."""
        if features:
            conn.executemany(
                "INSERT OR REPLACE INTO features (source, idx, density) VALUES (?, ?, ?)",
                features,
            )
            features.clear()
        if labels:
            conn.executemany(
                "INSERT OR REPLACE INTO labels (source, idx, method, text) VALUES (?, ?, ?, ?)",
                labels,
            )
            labels.clear()
        if embeddings:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (source, idx, embedding) VALUES (?, ?, ?)",
                embeddings,
            )
            embeddings.clear()
        if logits:
            conn.executemany(
                "INSERT OR REPLACE INTO logits (source, idx, direction, token, score) VALUES (?, ?, ?, ?, ?)",
                logits,
            )
            logits.clear()

    # --- Single lookups ---

    def get_label(
        self, feature_index: int,
        model_id: str, layer: int, hook: str, width: int | str,
        method: str = "label",
    ) -> str | None:
        """Get the label for a single feature."""
        source = self._ensure_source(model_id, layer, hook, width)
        row = self._get_conn().execute(
            "SELECT text FROM labels WHERE source = ? AND idx = ? AND method = ?",
            (source, feature_index, method),
        ).fetchone()
        return row[0] if row else None

    def get_labels(
        self, feature_indices: list[int],
        model_id: str, layer: int, hook: str, width: int | str,
        method: str = "label",
    ) -> dict[int, str]:
        """Batch lookup labels for multiple feature indices."""
        if not feature_indices:
            return {}
        source = self._ensure_source(model_id, layer, hook, width)
        placeholders = ",".join("?" * len(feature_indices))
        rows = self._get_conn().execute(
            f"SELECT idx, text FROM labels WHERE source = ? AND idx IN ({placeholders}) AND method = ?",
            [source, *feature_indices, method],
        ).fetchall()
        return {idx: text for idx, text in rows}

    def get_density(
        self, feature_index: int,
        model_id: str, layer: int, hook: str, width: int | str,
    ) -> float | None:
        """Get the density (activation frequency) for a single feature."""
        source = self._ensure_source(model_id, layer, hook, width)
        row = self._get_conn().execute(
            "SELECT density FROM features WHERE source = ? AND idx = ?",
            (source, feature_index),
        ).fetchone()
        return row[0] if row else None

    def get_densities(
        self, model_id: str, layer: int, hook: str, width: int | str,
    ) -> torch.Tensor:
        """Get a tensor of densities for all features.

        Returns a (width,) float tensor. Cached after first call.
        """
        key = self._resolve_key(model_id, layer, hook, width)
        if key in self._density_cache:
            return self._density_cache[key]

        source = self._ensure_source(model_id, layer, hook, width)
        n_features = _width_as_int(width)
        densities = torch.zeros(n_features, dtype=torch.float32)
        for idx, density in self._get_conn().execute(
            "SELECT idx, density FROM features WHERE source = ?", (source,)
        ):
            if idx < n_features:
                densities[idx] = density

        self._density_cache[key] = densities
        return densities

    def get_logits(
        self, feature_index: int,
        model_id: str, layer: int, hook: str, width: int | str,
    ) -> dict[str, list[tuple[str, float]]]:
        """Get top/bottom logits for a feature.

        Returns:
            Dict with "top" and "bottom" keys, each mapping to a list
            of (token, score) tuples sorted by score descending.
        """
        source = self._ensure_source(model_id, layer, hook, width)
        result: dict[str, list[tuple[str, float]]] = {"top": [], "bottom": []}
        rows = self._get_conn().execute(
            "SELECT direction, token, score FROM logits "
            "WHERE source = ? AND idx = ? ORDER BY score DESC",
            (source, feature_index),
        ).fetchall()
        for direction, token, score in rows:
            result[direction].append((token, score))
        return result

    def get_feature(
        self, feature_index: int,
        model_id: str, layer: int, hook: str, width: int | str,
    ) -> dict | None:
        """Get a reconstructed feature record from the structured tables.

        Returns a dict with density, label, logits, and whether an
        embedding is available. Does NOT include raw activation examples
        (those live in separate activation files).
        """
        source = self._ensure_source(model_id, layer, hook, width)
        conn = self._get_conn()

        row = conn.execute(
            "SELECT density FROM features WHERE source = ? AND idx = ?",
            (source, feature_index),
        ).fetchone()
        if not row:
            return None

        label = self.get_label(feature_index, model_id, layer, hook, width)
        logits = self.get_logits(feature_index, model_id, layer, hook, width)
        has_embedding = conn.execute(
            "SELECT 1 FROM embeddings WHERE source = ? AND idx = ?",
            (source, feature_index),
        ).fetchone() is not None

        return {
            "index": feature_index,
            "density": row[0],
            "label": label,
            "top_logits": logits["top"],
            "bottom_logits": logits["bottom"],
            "has_embedding": has_embedding,
        }

    # --- Embeddings ---

    def get_embeddings(
        self, model_id: str, layer: int, hook: str, width: int | str,
    ) -> tuple[torch.Tensor, list[int]]:
        """Load all explanation embeddings as a matrix.

        Returns:
            (embeddings, indices) where embeddings is a (N, 256) float tensor
            and indices is the list of corresponding feature indices.
            Cached after first call.
        """
        key = self._resolve_key(model_id, layer, hook, width)
        if key in self._embedding_cache:
            return self._embedding_cache[key]

        source = self._ensure_source(model_id, layer, hook, width)
        rows = self._get_conn().execute(
            "SELECT idx, embedding FROM embeddings WHERE source = ? ORDER BY idx",
            (source,),
        ).fetchall()

        if not rows:
            empty = (torch.zeros(0, _EMBEDDING_DIM), [])
            self._embedding_cache[key] = empty
            return empty

        indices = [r[0] for r in rows]
        vectors = [_unpack_embedding(r[1]) for r in rows]
        matrix = torch.tensor(vectors, dtype=torch.float32)

        result = (matrix, indices)
        self._embedding_cache[key] = result
        return result

    def find_similar_features(
        self, feature_index: int,
        model_id: str, layer: int, hook: str, width: int | str,
        k: int = 20,
    ) -> list[tuple[int, float, str]]:
        """Find features with semantically similar explanation labels.

        Uses cosine similarity between the stored 256-dim explanation
        embeddings. Only works for feature-to-feature similarity (we
        don't know the embedding model, so text queries aren't supported).

        Args:
            feature_index: The query feature index.
            model_id: Neuronpedia model ID.
            layer: Decoder layer index.
            hook: Hook type string.
            width: SAE width.
            k: Number of similar features to return.

        Returns:
            List of (feature_index, cosine_similarity, label) sorted
            by similarity descending. Excludes the query feature itself.
        """
        matrix, indices = self.get_embeddings(model_id, layer, hook, width)
        if matrix.shape[0] == 0:
            return []

        try:
            pos = indices.index(feature_index)
        except ValueError:
            return []

        query = matrix[pos].unsqueeze(0)  # (1, 256)
        sims = torch.nn.functional.cosine_similarity(query, matrix, dim=1)  # (N,)
        sims[pos] = -1.0  # exclude self

        topk = torch.topk(sims, k=min(k, sims.shape[0]))
        top_indices = [indices[i] for i in topk.indices.tolist()]
        labels_map = self.get_labels(top_indices, model_id, layer, hook, width)

        return [
            (idx, sim, labels_map.get(idx, "(unlabelled)"))
            for idx, sim in zip(top_indices, topk.values.tolist())
        ]

    # --- Write custom labels ---

    def write_labels(
        self, labels: dict[int, str],
        model_id: str, layer: int, hook: str, width: int | str,
        method: str = "label",
    ) -> None:
        """Write or replace labels for a given method.

        Args:
            labels: Dict mapping feature_index -> label text.
            model_id: Neuronpedia model ID (e.g. "gemma-3-4b-it").
            layer: Decoder layer index.
            hook: Hook type string (e.g. "resid_post").
            width: SAE width / d_sae.
            method: Labelling method name (e.g. "colour_probe", "custom").
        """
        source = self._ensure_source(model_id, layer, hook, width)
        conn = self._get_conn()
        conn.executemany(
            "INSERT OR REPLACE INTO labels (source, idx, method, text) VALUES (?, ?, ?, ?)",
            [(source, idx, method, text) for idx, text in labels.items()],
        )
        conn.commit()

    # --- Top-K labelling ---

    def label_top_k(
        self,
        feature_acts: torch.Tensor,
        model_id: str,
        layer: int,
        hook: str,
        width: int | str,
        k: int = 10,
        mask: torch.Tensor | None = None,
        method: str = "label",
    ) -> list[tuple[int, float, str]]:
        """Get top-K labelled features from a 1D activation tensor.

        Computes top-K from the tensor first (pure torch), then fetches
        labels only for those K indices from SQLite.

        Returns:
            List of (feature_index, activation_value, label) sorted by
            activation value descending.
        """
        acts = feature_acts.detach().float().cpu()
        if mask is not None:
            acts = torch.where(mask.cpu(), acts, torch.tensor(float("-inf")))

        topk = torch.topk(acts, k=min(k, acts.shape[0]))

        indices = []
        values = []
        for val, idx in zip(topk.values, topk.indices):
            if val.item() == float("-inf"):
                break
            indices.append(idx.item())
            values.append(val.item())

        labels_map = self.get_labels(indices, model_id, layer, hook, width, method)

        return [
            (idx, val, labels_map.get(idx, "(unlabelled)"))
            for idx, val in zip(indices, values)
        ]

    def label_top_k_per_token(
        self,
        feature_acts: torch.Tensor,
        model_id: str,
        layer: int,
        hook: str,
        width: int | str,
        k: int = 5,
        mask: torch.Tensor | None = None,
        method: str = "label",
    ) -> list[list[tuple[int, float, str]]]:
        """Get top-K labelled features for each token position.

        Args:
            feature_acts: Activation tensor of shape (seq_len, d_sae).
            model_id: Neuronpedia model ID (e.g. "gemma-3-4b-it").
            layer: Decoder layer index.
            hook: Hook type string (e.g. "resid_post").
            width: SAE width / d_sae.
            k: Number of top features per token.
            mask: Optional boolean tensor (d_sae,). Applied to all tokens.
            method: Labelling method to retrieve.

        Returns:
            List of length seq_len, each element a list of
            (feature_index, activation_value, label) tuples.
        """
        return [
            self.label_top_k(feature_acts[pos], model_id, layer, hook, width, k, mask, method)
            for pos in range(feature_acts.shape[0])
        ]

    @staticmethod
    def params_from_config(config: SAEConfig) -> tuple[str, int, str, str]:
        """Extract (model_id, layer, hook, width) from an SAEConfig.

        Returns:
            (neuronpedia_model_id, layer_index, hook_type_value, width) tuple
            for use with label lookup methods. Width is the string form
            (e.g. "16k") as stored on SAEConfig.
        """
        return config.neuronpedia_model_id, config.layer_index, config.hook_type.value, config.width
