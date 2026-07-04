"""Config-driven projection-fidelity (Mantel) evaluation.

Reads ``projection_fidelity_config.toml`` (next to this file by default, or a path
in ``ORRERY_PROJECTION_FIDELITY_CONFIG``), then for each configured collection it
loads the requested projections + item metadata from DuckDB and the original
embeddings from ChromaDB, builds pairwise-distance structures, and runs
:class:`ProjectionFidelityEvaluator` to score how faithfully each projection
preserves (a) the embedding geometry and (b) perceptual colour distance.

Run:
    uv run python -m interpretability_backend.evaluation.run_projection_fidelity

Run this with the backend stopped — DuckDB is single-writer.
"""

import logging
from pathlib import Path

import numpy as np

from interpretability_backend.backend.utils.duckdb_sync import _get_db as _get_duckdb
from interpretability_backend.backend.utils.embedding_loader import load_embeddings_for_ids
from interpretability_backend.evaluation.projection_fidelity import ProjectionFidelityEvaluator
from interpretability_backend.evaluation.utils import (
    load_config,
    resolve_config_path,
    write_results,
)

logger = logging.getLogger("orrery." + __name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "projection_fidelity_config.toml"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "projection_fidelity_results.json"


def _load_projections(duckdb, collection_name: str, projection_types: list[str]):
    """Load requested projections; return ``(ids, item_metadata, {ptype: coords})``.

    Uses the first successfully loaded projection's id order as canonical and
    aligns every other projection to it, dropping any that don't cover all ids.
    """
    canonical_ids: list[str] | None = None
    item_metadata: list[dict] | None = None
    coords_by_type: dict[str, np.ndarray] = {}

    for ptype in projection_types:
        data = duckdb.get_projection_data(collection_name, ptype)
        if not data:
            logger.warning("%s has no %s projection; skipping it", collection_name, ptype)
            continue
        if canonical_ids is None:
            canonical_ids = data["ids"]
            item_metadata = data["item_metadata"]
            coords_by_type[ptype] = np.asarray(data["coordinates"], dtype=np.float64)
        else:
            lut = dict(zip(data["ids"], data["coordinates"], strict=True))
            if not all(i in lut for i in canonical_ids):
                logger.warning(
                    "%s projection %s does not cover all ids; skipping it",
                    collection_name,
                    ptype,
                )
                continue
            coords_by_type[ptype] = np.asarray([lut[i] for i in canonical_ids], dtype=np.float64)
    return canonical_ids, item_metadata, coords_by_type


def evaluate_collection(
    duckdb,
    collection_name: str,
    projection_types: list[str],
    colour_field: str,
    k: int,
    n_perms: int,
    seed: int,
    sample_size: int,
) -> dict | None:
    """Score one collection's projections against embedding + colour references."""
    ids, item_metadata, coords_by_type = _load_projections(
        duckdb, collection_name, projection_types
    )
    if not ids or not coords_by_type:
        logger.warning("%s: no usable projections; skipping", collection_name)
        return None

    # Cost guard: subsample (pairwise distances are O(N^2)).
    n = len(ids)
    if n > sample_size:
        idx = np.sort(np.random.default_rng(seed).choice(n, sample_size, replace=False))
        ids = [ids[i] for i in idx]
        item_metadata = [item_metadata[i] for i in idx]
        coords_by_type = {p: c[idx] for p, c in coords_by_type.items()}
        logger.info("%s: subsampled %d -> %d items", collection_name, n, sample_size)

    ev = ProjectionFidelityEvaluator(k=k, n_perms=n_perms, seed=seed)

    references: dict[str, np.ndarray] = {}

    embeddings = load_embeddings_for_ids(collection_name, ids)
    if embeddings is not None:
        references["embedding (cosine)"] = ev.embedding_distances(embeddings)
    else:
        logger.warning("%s: embeddings unavailable; embedding reference skipped", collection_name)

    if colour_field:
        hexes = [m.get(colour_field) for m in item_metadata]
        try:
            references["colour (CIEDE2000)"] = ev.colour_distances(hexes)
        except ImportError:
            logger.warning(
                "%s: scikit-image not installed; colour reference skipped "
                "(install it or `uv sync` to enable)",
                collection_name,
            )
        except ValueError as e:
            logger.warning("%s: colour reference skipped (%s)", collection_name, e)

    if not references:
        logger.warning("%s: no reference structures available; skipping", collection_name)
        return None

    targets = {p: ev.projection_distances(c) for p, c in coords_by_type.items()}

    result = ev.evaluate(references=references, targets=targets, cross_reference=True)
    result["collection_name"] = collection_name
    return result


def _print_report(result: dict) -> None:
    """Pretty-print one collection's fidelity comparisons, grouped by reference."""
    print("\n" + "=" * 78)
    print(
        f"PROJECTION FIDELITY: {result['collection_name']}  "
        f"({result['n_items']} items, k={result['k']}, perms={result['n_perms']})"
    )
    print("=" * 78)

    def fmt(v, w=8):
        return f"{v:+.4f}".rjust(w) if isinstance(v, float) else "n/a".rjust(w)

    by_ref: dict[str, list[dict]] = {}
    for c in result["comparisons"]:
        by_ref.setdefault(c["reference"], []).append(c)

    for ref, comps in by_ref.items():
        kind = comps[0]["kind"]
        header = "baseline" if kind == "baseline" else "reference"
        print(f"\n  {header}: {ref}")
        print(f"    {'target':<22}{'global ρ':>10}{'kNN ρ':>10}{'perm z':>10}{'p_emp':>9}")
        for c in comps:
            p_emp = c["perm_empirical_p"]
            p_str = f"{p_emp:.3f}".rjust(9) if isinstance(p_emp, float) else "n/a".rjust(9)
            print(
                f"    {c['target']:<22}{fmt(c['global_rho'], 10)}{fmt(c['knn_rho'], 10)}"
                f"{fmt(c['perm_z'], 10)}{p_str}"
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    config_path = resolve_config_path("ORRERY_PROJECTION_FIDELITY_CONFIG", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    collections = config.get("collections", [])
    projection_types = config.get("projection_types", ["umap_3d", "pca_3d"])
    colour_field = config.get("colour_field", "colour_code")
    k = int(config.get("k", 10))
    n_perms = int(config.get("n_perms", 1000))
    seed = int(config.get("seed", 42))
    sample_size = int(config.get("sample_size", 2000))
    output_path = Path(config.get("output_path", DEFAULT_OUTPUT_PATH))

    if not collections:
        print(f"No collections listed in {config_path}. Add a `collections = [...]` entry.")
        return

    duckdb = _get_duckdb()
    if duckdb is None:
        print("DuckDB unavailable; cannot evaluate.")
        return

    results = []
    for collection_name in collections:
        result = evaluate_collection(
            duckdb,
            collection_name,
            projection_types,
            colour_field,
            k,
            n_perms,
            seed,
            sample_size,
        )
        if result:
            results.append(result)
            _print_report(result)

    write_results(output_path, results)


if __name__ == "__main__":
    main()
