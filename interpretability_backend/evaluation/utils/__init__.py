"""Shared helpers for the evaluation runners.

Small, dependency-free utilities factored out of the config-driven runners
(``run_evaluation.py``, ``run_projection_fidelity.py``) so the config-load /
path-resolution / results-write boilerplate lives in one place.
"""

from .runner_common import load_config, resolve_config_path, write_results

__all__ = ["load_config", "resolve_config_path", "write_results"]
