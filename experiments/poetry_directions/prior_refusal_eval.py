"""Generate unsteered + steered responses for a prior-labelled prompt set.

Reads ``resources/poetry/alignment_dataset_long_clean.tsv``, filters rows
where ``type == "prose"``, and for each prompt generates two responses:
one with no hooks (baseline) and one under additive steering with the
``poetry_prose`` experiment's chosen direction at the
``(layer, coefficient)`` the sweep selected. The output is appended row by
row (and flushed) so a crash mid-run preserves all completed rows; rerunning
skips any ``sample_id`` already present in the output TSV.

The output TSV is intended for manual labelling — for each prompt,
``response_unsteered`` shows what the unmodified model produces and
``response_steered`` shows what the steered model produces. The
``prior_safety_status`` column is the upstream label (``safe`` ⇒ original
model refused, ``unsafe`` ⇒ original model complied).

Run::

    uv run python -m interpret.experiments.poetry_directions.prior_refusal_eval

The experiment used for steering (default ``poetry_prose``), the input TSV
path, and the output filename are constants near the top — edit them to
point at a different experiment / dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from interpret.experiments.poetry_directions import PoetryConfig
from interpret.experiments.refusal_directions.select_direction import _additive_op
from interpret.experiments.refusal_directions.tokens import format_chat
from interpret.inference.gemma_pytorch import GemmaPytorchInference
from interpret.sae import HookManager

EXPERIMENT_NAME = "poetry_prose"
DATASET_PATH = Path("resources/poetry/alignment_dataset_long_clean.tsv")
PROMPT_TYPE_FILTER = "prose"
OUTPUT_FILENAME = "prior_refusal_steered_responses.tsv"

OUTPUT_COLUMNS = [
    "sample_id",
    "hazard",
    "prompt",
    "prior_safety_status",
    "response_unsteered",
    "response_steered",
]


def _load_direction(cfg: PoetryConfig) -> tuple[torch.Tensor, int, float]:
    """Return (direction, source_layer, coefficient) from sweep artifacts."""
    direction_path = cfg.output_dir / "direction.pt"
    metadata_path = cfg.output_dir / "direction_metadata.json"
    if not (direction_path.exists() and metadata_path.exists()):
        raise FileNotFoundError(
            f"Missing direction artifacts for {cfg.name!r}: expected "
            f"{direction_path} and {metadata_path}. Run the sweep first."
        )
    direction = torch.load(direction_path, map_location="cpu").to(torch.float32)
    metadata = json.loads(metadata_path.read_text())
    return direction, int(metadata["layer"]), float(metadata["coefficient"])


def _already_done(out_path: Path) -> set[str]:
    """Sample_ids already present in the output TSV."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    done: set[str] = set()
    with out_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("sample_id")
            if sid:
                done.add(str(sid))
    return done


def _append_row(out_path: Path, row: dict) -> None:
    """Append one row to a TSV (writing the header on first write); flush + close per call."""
    file_has_header = out_path.exists() and out_path.stat().st_size > 0
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        if not file_has_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main() -> None:
    cfg = PoetryConfig(name=EXPERIMENT_NAME)
    direction, source_layer, coefficient = _load_direction(cfg)
    print(
        f"steering with experiment={EXPERIMENT_NAME} "
        f"layer={source_layer} coefficient={coefficient:+.2f} "
        f"|v|={direction.norm().item():.3f}"
    )

    out_path = cfg.output_dir / OUTPUT_FILENAME
    print(f"output → {out_path}")
    done = _already_done(out_path)
    print(f"resuming with {len(done)} sample_ids already in the output")

    df = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        usecols=["sample_id", "type", "hazard", "prompt", "final_safety_status"],
    )
    df = df[df["type"] == PROMPT_TYPE_FILTER]
    df = df.dropna(subset=["sample_id", "prompt"])
    df["sample_id"] = df["sample_id"].astype(str)
    df = df[~df["sample_id"].isin(done)]
    n_remaining = len(df)
    print(f"{n_remaining} remaining {PROMPT_TYPE_FILTER!r} prompts to process")

    if n_remaining == 0:
        return

    wrapper = GemmaPytorchInference(cfg.model_name)
    layers = wrapper.model.model.layers

    manager = HookManager()
    manager.add_steering([_additive_op(direction, source_layer, coeff=coefficient)])

    for _, item in tqdm(df.iterrows(), total=n_remaining, desc="prior_refusal_eval"):
        sid = item["sample_id"]
        prompt = item["prompt"]
        formatted = format_chat(wrapper, prompt)

        response_unsteered = wrapper.generate_from_template(
            formatted, output_len=cfg.max_new_tokens, temperature=None
        )
        with manager.session(layers):
            response_steered = wrapper.generate_from_template(
                formatted, output_len=cfg.max_new_tokens, temperature=None
            )

        _append_row(
            out_path,
            {
                "sample_id": sid,
                "hazard": item.get("hazard", "") or "",
                "prompt": prompt,
                "prior_safety_status": item.get("final_safety_status", "") or "",
                "response_unsteered": response_unsteered,
                "response_steered": response_steered,
            },
        )


if __name__ == "__main__":
    main()
