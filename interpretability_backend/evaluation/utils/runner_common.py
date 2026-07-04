"""Config + I/O boilerplate shared by the evaluation runners.

Both ``run_evaluation.py`` and ``run_projection_fidelity.py`` resolve a TOML
config path (with an env-var override), load it, and write a JSON results list.
Those three steps are identical across runners and live here to avoid
duplication.
"""

import json
import os
import tomllib
from pathlib import Path


def resolve_config_path(env_var: str, default: Path) -> Path:
    """Path from ``env_var`` if set, else ``default``."""
    return Path(os.getenv(env_var, default))


def load_config(path: Path) -> dict:
    """Load a TOML config (stdlib ``tomllib``, no new dependency)."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_results(path: Path, results: list) -> None:
    """Write ``results`` as pretty JSON and print a one-line confirmation."""
    path = Path(path)
    path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} result(s) to {path}")
