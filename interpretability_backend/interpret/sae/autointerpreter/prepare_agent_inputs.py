"""Bridges Stage 2 output into the AgentSystem input folders.

- ``write_label_inputs`` — copies ``topk/feature_*.json`` into the label
  task's ``input_folder``. The LabelInterpreter agent reads these and
  produces ``{feature_index, short_name, explanation, polarity}``.
- ``write_evaluator_inputs`` — merges the agent-produced labels with the
  Stage 2 linspace samples (stripping ``_true_activations``) and writes
  them into the eval task's ``input_folder``. Also records the A/B split
  decision (zero-fraction hint on/off per feature) so the scorer can slice
  correlations by arm.

Filename tagging (global queue dirs only):
  Files written into the shared agent queue dirs are prefixed
  ``{tag}__feature_NNNNNN.json`` so chained configs cannot collide on the
  same feature index across different SAEs (e.g. feature 1567 may be in
  both an L9 w16k draw and an L29 w16k draw — without tagging the second
  config's label would be skipped because init_jobs sees the prior
  config's output file with the same name). Per-store ``topk/``,
  ``linspace/``, ``labels/``, and ``evaluator/`` dirs keep clean
  ``feature_NNNNNN.json`` names — the tag never reaches the persistent
  archive; the runner strips it when syncing global → per-store.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from interpret.sae.autointerpreter.config import AgentStageConfig

TAG_SEP = "__"


def tagged_name(tag: str, base: str) -> str:
    """Prepend a per-config tag to a feature filename.

    ``tagged_name("L9_w16k", "feature_001567.json")`` → ``"L9_w16k__feature_001567.json"``.
    Empty tag returns ``base`` unchanged (for back-compat with untagged paths).
    """
    if not tag:
        return base
    return f"{tag}{TAG_SEP}{base}"


def strip_tag(name: str) -> tuple[str, str]:
    """Split ``"tag__feature_NNNNNN.json"`` into ``(tag, "feature_NNNNNN.json")``.

    Returns ``("", name)`` when no tag separator is present. The separator is
    a double underscore so single-underscore-containing tags survive (e.g.
    ``"L29_w65k_resid_post_max_prefill"``).
    """
    if TAG_SEP not in name:
        return ("", name)
    tag, _, base = name.partition(TAG_SEP)
    return (tag, base)


def _parse_feature_idx(filename: str) -> int:
    """Parse the feature index from a (possibly tagged) filename."""
    _, base = strip_tag(Path(filename).name)
    stem = Path(base).stem
    return int(stem.split("_")[-1])


class AgentInputWriter:
    """Populates the AgentSystem input folders from a Stage 2 run directory."""

    AB_SPLIT_FILE = "ab_split.parquet"

    def __init__(
        self,
        run_dir: Path,
        agents: AgentStageConfig,
        tag: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.agents = agents
        # Per-config tag used to prefix global-queue filenames. Defaults to
        # the run dir's basename, which is unique across chained configs
        # (e.g. ``L29_w16k_resid_post_max_prefill`` vs ``L29_w16k_resid_post``).
        self.tag = tag if tag is not None else self.run_dir.name

    # ── Stage 3 (label) ──────────────────────────────────────────────────

    def write_label_inputs(self) -> int:
        src = self.run_dir / "topk"
        dst = Path(self.agents.label_input_dir)
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.glob("feature_*.json")):
            shutil.copy2(f, dst / tagged_name(self.tag, f.name))
            n += 1
        return n

    # ── Stage 4 (eval) ───────────────────────────────────────────────────

    def _show_zero_hint(self, feature_idx: int) -> bool:
        mode = self.agents.show_zero_fraction_to_evaluator
        if mode == "on":
            return True
        if mode == "off":
            return False
        # Deterministic A/B split: half of features see the hint.
        digest = hashlib.md5(f"{feature_idx}".encode()).digest()
        return digest[0] % 2 == 0

    def write_evaluator_inputs(self) -> dict:
        linspace_dir = self.run_dir / "linspace"
        # Always read labels from the per-store ``labels/`` dir. The runner
        # ensures it is populated (with clean, untagged filenames) by the
        # label-stage sync; this keeps eval-input pairing free of cross-config
        # collisions even when configs share global queue dirs.
        labels_dir = self.run_dir / "labels"
        dst = Path(self.agents.eval_input_dir)
        dst.mkdir(parents=True, exist_ok=True)

        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"label results folder missing: {labels_dir}. "
                "Run the label stage first."
            )

        ab_records: list[dict] = []
        n_written = 0
        n_skipped_no_label = 0

        for linspace_path in sorted(linspace_dir.glob("feature_*.json")):
            feature_idx = _parse_feature_idx(linspace_path.name)
            label_path = labels_dir / linspace_path.name
            if not label_path.exists():
                n_skipped_no_label += 1
                continue

            linspace = json.loads(linspace_path.read_text(encoding="utf-8"))
            label_data = json.loads(label_path.read_text(encoding="utf-8"))
            show_hint = self._show_zero_hint(feature_idx)

            payload = {
                "feature_index": feature_idx,
                "layer": linspace.get("layer"),
                "hook": linspace.get("hook"),
                "width": linspace.get("width"),
                "short_name": label_data.get("short_name", ""),
                "explanation": label_data.get("explanation", ""),
                "polarity": label_data.get("polarity"),
                "samples": [
                    {k: v for k, v in s.items() if k != "row_idx"}
                    for s in linspace.get("samples", [])
                ],
            }
            if show_hint:
                payload["zero_fraction"] = linspace.get("zero_fraction")

            (dst / tagged_name(self.tag, linspace_path.name)).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            ab_records.append(
                {"feature_idx": feature_idx, "zero_hint_shown": show_hint},
            )
            n_written += 1

        self._write_ab_split(ab_records)
        return {
            "n_written": n_written,
            "n_skipped_no_label": n_skipped_no_label,
        }

    def _write_ab_split(self, records: list[dict]) -> None:
        if not records:
            return
        out = self.run_dir / self.AB_SPLIT_FILE
        df = pd.DataFrame(records)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
