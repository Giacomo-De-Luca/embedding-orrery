"""Build graph bundles; all derivation options live in the selected JSON config."""

import argparse
from pathlib import Path

from interpretability_backend.backend.utils.graph_datasets.pipeline import GraphBuildRunner


class GraphBuildCommand:
    @staticmethod
    def run() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--config", type=Path,
                            default=Path(__file__).resolve().parents[1] / "config/graphs/acl_document.json")
        GraphBuildRunner(parser.parse_args().config).run()


if __name__ == "__main__":
    GraphBuildCommand.run()
