"""Stage-level cache for the probing pipeline.

Cached artefacts are stored as `<stem>.pt` files with a sibling
`<stem>.yaml` sidecar that records the full config + metadata.

Cache hit  = both files exist AND the sidecar's `config` block equals
             the current config (after enum/Path normalisation).
Cache miss = either file missing.
Mismatch   = both files exist BUT sidecar disagrees with current config.
             We raise `CacheMismatchError` rather than silently overwriting.
             The user must delete the stale `.pt` + `.yaml` to recompute.

Filenames are produced by each config's `cache_filename()` method (the
stem only — no extension). They are human-readable so a user can browse
`<output>/.cache/` and immediately see what each entry represents.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from collections.abc import Callable
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml


class CacheMismatchError(RuntimeError):
    """A cache file exists but its sidecar disagrees with the current config."""


def _normalise(obj: Any) -> Any:
    """Recursively convert a dataclass / enum / Path tree into a plain dict."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _normalise(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, dict):
        return {k: _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v) for v in obj]
    return obj


def _diff_summary(saved: Any, current: Any, prefix: str = "") -> list[str]:
    """Produce a short list of differences between two normalised configs."""
    if saved == current:
        return []
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return [f"{prefix}: saved={saved!r}  current={current!r}"]
    diffs: list[str] = []
    for k in sorted(set(saved) | set(current)):
        sub = f"{prefix}.{k}" if prefix else k
        if k not in saved:
            diffs.append(f"{sub}: <missing in saved>  current={current[k]!r}")
        elif k not in current:
            diffs.append(f"{sub}: saved={saved[k]!r}  <missing in current>")
        elif saved[k] != current[k]:
            diffs.extend(_diff_summary(saved[k], current[k], prefix=sub))
    return diffs


class StageCache:
    """Read/write stage outputs with readable names + YAML sidecars."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def paths(self, stem: str) -> tuple[Path, Path]:
        """Return the (.pt, .yaml) pair for a stem."""
        return self.cache_dir / f"{stem}.pt", self.cache_dir / f"{stem}.yaml"

    def get(self, stem: str, config: Any) -> Path | None:
        """Return cached `.pt` path if present + sidecar matches; else None.

        Raises `CacheMismatchError` if both files exist but the sidecar
        disagrees with the current config.
        """
        pt_path, yaml_path = self.paths(stem)
        if not pt_path.exists() or not yaml_path.exists():
            return None
        with open(yaml_path) as f:
            sidecar = yaml.safe_load(f) or {}
        saved = sidecar.get("config", {})
        current = _normalise(config)
        if saved != current:
            diffs = _diff_summary(saved, current)
            details = "\n  ".join(diffs[:10])
            more = (
                f"\n  ... (+{len(diffs) - 10} more)" if len(diffs) > 10 else ""
            )
            raise CacheMismatchError(
                f"Cache sidecar at {yaml_path} disagrees with the current "
                f"config:\n  {details}{more}\n"
                f"Delete {pt_path.name} + {yaml_path.name} from "
                f"{self.cache_dir} to recompute.",
            )
        return pt_path

    def put(
        self,
        stem: str,
        config: Any,
        write_fn: Callable[[Path], None],
    ) -> Path:
        """Write the artefact + sidecar atomically.

        `write_fn(path)` is called with a temporary path; we rename to the
        final location only on success.
        """
        pt_path, yaml_path = self.paths(stem)
        tmp_path = pt_path.with_suffix(".pt.tmp")
        try:
            write_fn(tmp_path)
            os.replace(tmp_path, pt_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        sidecar = {
            "config": _normalise(config),
            "meta": {
                "torch_version": str(torch.__version__),
                "python_version": (
                    f"{sys.version_info.major}.{sys.version_info.minor}"
                ),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(sidecar, f, default_flow_style=False, sort_keys=True)
        return pt_path
