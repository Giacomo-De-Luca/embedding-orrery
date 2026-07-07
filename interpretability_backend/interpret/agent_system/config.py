"""
Task configuration loader for the agent job queue framework.

Each task is defined by a JSON file in a tasks directory. Task JSONs are
looked up in order: ``$AGENT_TASKS_DIR`` (if set), the toolkit's own
``tasks/`` next to this file, then ``<PROJECT_ROOT>/scripts/agent_tasks``
(project-side tasks that live outside the toolkit).

Paths in the config are resolved relative to PROJECT_ROOT, which defaults
to two levels above this file (the repo root when the toolkit lives at
``<root>/interpret/agent_system``) and can be overridden with the
``AGENT_PROJECT_ROOT`` environment variable (e.g. for a standalone toolkit
checkout, where the root is one level up).

Path templating
---------------
Path fields (`input_folder`, `output_folder`, `jobs_folder`) may contain a
``{variant}`` placeholder. Callers pass ``variant="..."`` to ``load_config``
to fan a single task config out across multiple isolated working dirs (for
example, one (SAE × model) cell per autointerpret batch). An empty
variant collapses ``//`` cleanly via pathlib normalization so legacy task
JSONs without the placeholder keep working unchanged.

Model override
--------------
``model_override`` lets the launcher pin a specific model for a single
batch without forking the task JSON. ``None`` (the default) preserves
the value set inside the JSON.
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.getenv("AGENT_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
).expanduser()
TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def _tasks_search_dirs():
    """Task-JSON lookup order: env override, toolkit tasks/, project tasks."""
    dirs = []
    env_dir = os.getenv("AGENT_TASKS_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    dirs.append(TASKS_DIR)
    dirs.append(PROJECT_ROOT / "scripts" / "agent_tasks")
    return dirs


def find_task_config(task_name):
    """Return the path of ``<task_name>.json``, searching all task dirs."""
    for d in _tasks_search_dirs():
        candidate = d / f"{task_name}.json"
        if candidate.exists():
            return candidate
    searched = ", ".join(str(d) for d in _tasks_search_dirs())
    raise FileNotFoundError(
        f"Task config not found: {task_name}.json (searched: {searched})"
    )

REQUIRED_FIELDS = {"task_name", "agent", "input_folder", "jobs_folder"}
DEFAULTS = {
    "model": "opus",
    "output_folder": None,
    "stale_timeout_minutes": 30,
    "on_complete": None,
}


def _resolve_path(p):
    """Resolve a path: absolute stays absolute, relative joins with PROJECT_ROOT."""
    if p is None:
        return None
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _substitute_variant(template, variant):
    """Replace ``{variant}`` in a path template; empty variant collapses ``//``."""
    if template is None:
        return None
    return str(template).replace("{variant}", variant or "")


def load_config(task_name, variant=None, model_override=None):
    """
    Load and validate a task config from tasks/{task_name}.json.

    ``variant`` is substituted into the ``{variant}`` placeholder in
    ``input_folder``, ``output_folder`` and ``jobs_folder`` (use it to
    fan one task across many isolated working dirs). ``model_override``
    pins the agent model for one invocation without editing the JSON.

    Returns a dict with all fields populated (defaults filled in,
    paths resolved against PROJECT_ROOT).
    """
    config_path = find_task_config(task_name)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Strip _comment keys (used for documentation in example configs)
    config = {k: v for k, v in config.items() if not k.startswith("_")}

    validate_config(config)

    # Apply defaults
    for key, default in DEFAULTS.items():
        config.setdefault(key, default)

    # Substitute {variant} BEFORE path resolution — keeps the templated
    # form invisible to the rest of the system.
    for key in ("input_folder", "output_folder", "jobs_folder"):
        config[key] = _substitute_variant(config[key], variant)

    # Resolve paths
    for key in ("input_folder", "output_folder", "jobs_folder"):
        config[key] = _resolve_path(config[key])

    if model_override:
        config["model"] = model_override

    # Surface variant downstream so consumers (logs, manifest) can record it.
    config["variant"] = variant or ""

    return config


def validate_config(config):
    """Check that all required fields are present."""
    missing = REQUIRED_FIELDS - set(config.keys())
    if missing:
        raise ValueError(f"Task config missing required fields: {missing}")
