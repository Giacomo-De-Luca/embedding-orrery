"""Config-driven comparison of topic-reduction strategies over stored collections.

Reads ``reduction_comparison.toml`` (next to this file by default, or a path in
the ``ORRERY_REDUCTION_COMPARISON_CONFIG`` env var), then for each configured
collection cuts its base topics with every strategy (``ReductionComparison``)
and writes the scores as JSON plus a Markdown report next to it. Nothing is
stored in the database.

Run with the backend stopped (DuckDB is single-writer), or point
``duckdb_path`` / ``chroma_path`` at a copy (e.g. of the seed snapshot):
    uv run python -m interpretability_backend.evaluation.run_reduction_comparison
"""

import logging
from functools import partial
from pathlib import Path

from interpretability_backend.backend.clients.duckdb_client import DuckDBClient
from interpretability_backend.backend.services.topic_centroid_service import TopicCentroidService
from interpretability_backend.backend.services.topic_merge_tree_service import (
    TopicMergeTreeService,
)
from interpretability_backend.backend.services.topic_reduction_service import (
    TopicReductionService,
)
from interpretability_backend.backend.utils.embedding_loader import iter_embedding_batches
from interpretability_backend.evaluation.reduction_comparison import ReductionComparison
from interpretability_backend.evaluation.utils.runner_common import (
    load_config,
    resolve_config_path,
    write_results,
)

logger = logging.getLogger("orrery." + __name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "reduction_comparison.toml"
DEFAULT_OUTPUT_PATH = Path(__file__).parent / "reduction_comparison_results.json"
REPORT_COLUMNS = (
    ("case", "Case"),
    ("n_topics", "Topics"),
    ("largest_share", "Largest"),
    ("size_entropy", "Size entropy"),
    ("noise_share", "Noise"),
    ("absorbed_noise", "Absorbed"),
    ("topic_diversity", "Diversity"),
    ("coherence_cv", "C_v"),
    ("coherence_umass", "U_Mass"),
    ("silhouette_cluster_space", "Silhouette"),
    ("nmi", "NMI"),
    ("ari", "ARI"),
)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(results: list[dict]) -> str:
    """One table of scores per collection, then the strategy agreement."""
    lines = ["# Topic reduction comparison", ""]
    for result in results:
        lines += [
            f"## {result['collection']}",
            "",
            f"{result['base_topics']} base topics ({result['base_level']} level), "
            f"{result['items']} items.",
            "",
            "| " + " | ".join(title for _, title in REPORT_COLUMNS) + " |",
            "|" + "---|" * len(REPORT_COLUMNS),
        ]
        for row in result["rows"]:
            if "skipped" in row:
                lines.append(f"| {row['case']} | skipped: {row['skipped']} |")
            else:
                lines.append(
                    "| " + " | ".join(_cell(row.get(key)) for key, _ in REPORT_COLUMNS) + " |"
                )
        if result["agreement"]:
            lines += ["", "Agreement between strategies (ARI, same topic count):", ""]
            lines += ["| Topics | A | B | ARI |", "|---|---|---|---|"]
            for pair in result["agreement"]:
                lines.append(
                    f"| {pair['n_topics']} | {pair['a']} | {pair['b']} | {_cell(pair['ari'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    config_path = resolve_config_path("ORRERY_REDUCTION_COMPARISON_CONFIG", DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    db = DuckDBClient(db_path=config.get("duckdb_path") or None)
    chroma_path = config.get("chroma_path")
    vector_batches = (
        partial(iter_embedding_batches, chroma_path=Path(chroma_path))
        if chroma_path
        else iter_embedding_batches
    )
    trees = TopicMergeTreeService(
        db, centroid_service=TopicCentroidService(duckdb=db, vector_batches=vector_batches)
    )
    comparison = ReductionComparison(
        db,
        TopicReductionService(db, trees=trees),
        metrics=set(config["metrics"]) if config.get("metrics") else None,
        sample_size=int(config.get("sample_size", 10000)),
        n_keywords=int(config.get("n_keywords", 10)),
    )
    cases = comparison.cases(
        strategies=list(config.get("strategies", ["average", "ward", "density"])),
        n_topics=[int(n) for n in config.get("n_topics", [5])],
        distance_thresholds=[float(t) for t in config.get("distance_thresholds", [])],
        legacy_auto=bool(config.get("legacy_auto", True)),
        absorb_noise=bool(config.get("absorb_noise", True)),
    )
    ground_truth = config.get("ground_truth", {})

    results = []
    try:
        for collection in config.get("collections", []):
            print(f"Comparing reductions on {collection}...")
            try:
                results.append(comparison.compare(collection, cases, ground_truth.get(collection)))
            except ValueError as exc:
                logger.error("Skipping %s: %s", collection, exc)
                print(f"  skipped: {exc}")
    finally:
        db.close()

    output_path = Path(config.get("output_path") or DEFAULT_OUTPUT_PATH)
    write_results(output_path, results)
    report_path = output_path.with_suffix(".md")
    report_path.write_text(markdown_report(results))
    print(f"Wrote the report to {report_path}")


if __name__ == "__main__":
    main()
