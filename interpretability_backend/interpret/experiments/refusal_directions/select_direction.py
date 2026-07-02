"""Three-metric sweep + best-direction selection.

Adapted from `references/refusal_direction/pipeline/submodules/select_direction.py`.

For every candidate ``(intermediate, position, layer)``:

* **bypass score**  — refusal score on harmful_val under the bypass
  intervention. ``cfg.bypass_mode == "ablation"`` uses three-site projection
  ablation at every ``(layer, hook_type)`` for
  ``hook_type in {RESID_POST, ATTN_OUT, MLP_OUT}`` (Arditi's recipe).
  ``cfg.bypass_mode == "actadd"`` uses a single additive op at the source
  layer with ``cfg.actadd_bypass_coeff`` — required when full ablation
  collapses the residual stream (Gemma-3, Qwen3);
* **induce score** — refusal score on harmless_val under additive steering at
  the source layer (RESID_POST);
* **KL on harmless** — KL between baseline last-position logits and intervened
  last-position logits over harmless_val (under the same bypass intervention).

Filtering follows the paper: drop the top ``prune_layer_pct`` of layers, drop
KL > kl_threshold, drop induce score < induce_refusal_threshold. The remaining
direction with the lowest refusal score under ablation wins.

Model access is via a ``DirectionModel`` adapter, so the same sweep runs on
Gemma or Qwen. The scoring + steering primitives live in
``interpret.experiments.directions_common`` and are re-exported here for
backward compatibility.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Shared primitives (re-exported: existing imports of these names from this
# module — e.g. poetry sweep.py, prior_refusal_eval.py — keep working).
from interpret.experiments.directions_common import (
    DirectionModel,
    _ablation_ops,
    _additive_op,
    _bypass_ops,
    _kl_div,
    _make_manager,
    _refusal_score,
    _score_dataset,
)
from interpret.experiments.refusal_directions.config import RefusalConfig

__all__ = [
    "select_direction",
    "_ablation_ops",
    "_additive_op",
    "_bypass_ops",
    "_make_manager",
    "_score_dataset",
    "_refusal_score",
    "_kl_div",
]

_ZERO_NORM_TOL = 1e-8


def _eval_signature(config: RefusalConfig) -> dict:
    """Identify the cfg knobs that change what the sweep computes.

    Used to invalidate cached ``direction_evaluations.json`` when any of these
    change. Plotting / threshold knobs (``kl_threshold``, ``prune_layer_pct``,
    ``induce_refusal_threshold``) are NOT part of the signature — they only
    affect the filter, so the cache survives threshold-tuning reruns.
    """
    return {
        "bypass_mode": config.bypass_mode,
        "actadd_bypass_coeff": config.actadd_bypass_coeff,
        "n_val": config.n_val,
        "intermediates": list(config.intermediates),
        "refusal_token_ids": list(config.refusal_token_ids),
        "seed": config.seed,
    }


def _load_cached_evals(
    out_dir: Path, signature: dict
) -> list[dict] | None:
    """Return cached evaluation rows if the sidecar signature matches; else None."""
    evals_path = out_dir / "direction_evaluations.json"
    meta_path = out_dir / "direction_evaluations_meta.json"
    if not (evals_path.exists() and meta_path.exists()):
        return None
    with meta_path.open() as f:
        cached_sig = json.load(f)
    if cached_sig != signature:
        print(
            "[select_direction] cached evaluations don't match cfg "
            f"(diff: cached={cached_sig} vs current={signature}); will re-sweep"
        )
        return None
    with evals_path.open() as f:
        rows = json.load(f)
    print(
        f"[select_direction] reusing cached {evals_path} "
        f"({len(rows)} candidates; bypass_mode={signature['bypass_mode']}). "
        "Plots and direction selection re-derive from this cache."
    )
    return rows


def _save_evals_with_meta(
    out_dir: Path, rows: list[dict], signature: dict
) -> None:
    """Atomically write both the rows and the signature sidecar."""
    with (out_dir / "direction_evaluations.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (out_dir / "direction_evaluations_meta.json").open("w") as f:
        json.dump(signature, f, indent=2)


def _plot_refusal_scores(
    scores: torch.Tensor,
    baseline: float | None,
    pos_labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Direct port of reference plot_refusal_scores."""
    n_pos, n_layer = scores.shape
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(-n_pos, 0):
        ax.plot(
            range(n_layer),
            scores[i].cpu().numpy(),
            label=f"{i}: {pos_labels[i]!r}",
        )
    if baseline is not None:
        ax.axhline(y=baseline, color="black", linestyle="--")
        ax.annotate(
            "Baseline",
            xy=(1, baseline),
            xytext=(8, 10),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            ha="right",
            va="center",
        )
    ax.set_title(title)
    ax.set_xlabel("Layer source of direction")
    ax.set_ylabel("Refusal score")
    ax.legend(title="Position", loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _run_sweep(
    model: DirectionModel,
    candidates: dict[str, torch.Tensor],
    harmful_val: list[str],
    harmless_val: list[str],
    baseline_harmful: torch.Tensor,
    baseline_harmless: torch.Tensor,
    baseline_harmless_logits: list[torch.Tensor],
    pos_labels: list[str],
    config: RefusalConfig,
    out_dir: Path,
) -> list[dict]:
    """Run the (intermediate, position, layer) sweep and write per-intermediate plots.

    Returns the flat list of evaluation rows. Side effects: writes
    ``ablation_scores_<intermediate>.png`` / ``actadd_scores_<intermediate>.png``
    / ``kl_div_scores_<intermediate>.png`` into ``out_dir`` for each intermediate.
    """
    n_layers = config.n_layers
    refusal_toks = config.refusal_token_ids
    all_rows: list[dict] = []

    for intermediate, candidate_tensor in candidates.items():
        n_pos, layer_count, _ = candidate_tensor.shape
        assert layer_count == n_layers, (
            f"candidate shape mismatch: {candidate_tensor.shape} vs n_layers={n_layers}"
        )

        ablation_refusal = torch.full((n_pos, n_layers), float("nan"))
        induce_refusal = torch.full((n_pos, n_layers), float("nan"))
        kl_scores = torch.full((n_pos, n_layers), float("nan"))

        for source_pos in range(n_pos):
            for source_layer in tqdm(
                range(n_layers),
                desc=f"sweep[{config.bypass_mode}] {intermediate} pos={source_pos - n_pos}",
            ):
                direction = candidate_tensor[source_pos, source_layer].to(
                    torch.float32
                )
                if direction.norm().item() < _ZERO_NORM_TOL:
                    # Layer 0's pre_attn (and any other position whose mean
                    # diff happens to vanish) yields a zero direction. Score
                    # as NaN so the filter discards it.
                    all_rows.append(
                        {
                            "intermediate": intermediate,
                            "position": source_pos - n_pos,
                            "layer": source_layer,
                            "refusal_score": float("nan"),
                            "induce_score": float("nan"),
                            "kl_score": float("nan"),
                        }
                    )
                    continue

                # Bypass: cfg.bypass_mode picks ablation (every layer × 3 sites,
                # paper recipe) or actadd (single op at source_layer, used when
                # full ablation collapses the residual stream — Gemma-3, Qwen3).
                bypass_mgr = _make_manager(
                    _bypass_ops(
                        direction,
                        n_layers,
                        source_layer,
                        config.bypass_mode,
                        actadd_coeff=config.actadd_bypass_coeff,
                    )
                )
                ablated_scores, _ = _score_dataset(
                    model, harmful_val, bypass_mgr, refusal_toks
                )
                ablation_refusal[source_pos, source_layer] = float(
                    sum(ablated_scores) / max(len(ablated_scores), 1)
                )

                # KL on harmless under the same intervention.
                _, ablated_harmless_logits = _score_dataset(
                    model, harmless_val, bypass_mgr, refusal_toks
                )
                kl_vals = [
                    _kl_div(b, a)
                    for b, a in zip(baseline_harmless_logits, ablated_harmless_logits)
                ]
                kl_scores[source_pos, source_layer] = float(
                    sum(kl_vals) / max(len(kl_vals), 1)
                )

                # Induce: additive at the source layer, score on harmless_val.
                induce_mgr = _make_manager(
                    [_additive_op(direction, source_layer, coeff=1.0)]
                )
                induce_scores, _ = _score_dataset(
                    model, harmless_val, induce_mgr, refusal_toks
                )
                induce_refusal[source_pos, source_layer] = float(
                    sum(induce_scores) / max(len(induce_scores), 1)
                )

                all_rows.append(
                    {
                        "intermediate": intermediate,
                        "position": source_pos - n_pos,
                        "layer": source_layer,
                        "refusal_score": ablation_refusal[
                            source_pos, source_layer
                        ].item(),
                        "induce_score": induce_refusal[
                            source_pos, source_layer
                        ].item(),
                        "kl_score": kl_scores[source_pos, source_layer].item(),
                    }
                )

        _plot_refusal_scores(
            ablation_refusal,
            baseline_harmful.mean().item(),
            pos_labels,
            f"Bypass ({config.bypass_mode}) on harmful — {intermediate}",
            out_dir / f"ablation_scores_{intermediate}.png",
        )
        _plot_refusal_scores(
            induce_refusal,
            baseline_harmless.mean().item(),
            pos_labels,
            f"Activation addition on harmless — {intermediate}",
            out_dir / f"actadd_scores_{intermediate}.png",
        )
        _plot_refusal_scores(
            kl_scores,
            0.0,
            pos_labels,
            f"KL divergence under bypass ({config.bypass_mode}) — {intermediate}",
            out_dir / f"kl_div_scores_{intermediate}.png",
        )

    return all_rows


def _filter(
    refusal_score: float,
    induce_score: float,
    kl_score: float,
    layer: int,
    n_layers: int,
    config: RefusalConfig,
) -> bool:
    """Reference filter — returns True to discard the direction."""
    if any(math.isnan(s) for s in (refusal_score, induce_score, kl_score)):
        return True
    if layer >= int(n_layers * (1.0 - config.prune_layer_pct)):
        return True
    if kl_score > config.kl_threshold:
        return True
    if induce_score < config.induce_refusal_threshold:
        return True
    return False


def select_direction(
    model: DirectionModel,
    candidates: dict[str, torch.Tensor],
    harmful_val: list[str],
    harmless_val: list[str],
    pos_labels: list[str],
    config: RefusalConfig,
) -> dict:
    """Run the full sweep over `candidates` and pick the best direction.

    `candidates[intermediate]` has shape ``(n_pos, n_layers, d_model)``.

    Returns a dict with keys ``intermediate``, ``position``, ``layer``,
    ``direction`` (the chosen 1-D tensor) and writes plots + JSON outputs to
    ``config.select_dir``.
    """
    out_dir = config.select_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    signature = _eval_signature(config)
    direction_path = config.output_dir / "direction.pt"
    metadata_path = config.output_dir / "direction_metadata.json"
    if direction_path.exists() and metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)
        cached_sig = metadata.get("eval_signature")
        if cached_sig == signature:
            print(f"[select_direction] reusing cached {direction_path}")
            return {
                **{k: v for k, v in metadata.items() if k != "eval_signature"},
                "direction": torch.load(direction_path, map_location="cpu"),
            }
        print(
            f"[select_direction] cached direction.pt was selected under "
            f"different cfg (cached={cached_sig}, current={signature}); "
            "re-selecting from cached evaluations if available."
        )

    n_layers = config.n_layers
    refusal_toks = config.refusal_token_ids

    cached_rows = _load_cached_evals(out_dir, signature)
    if cached_rows is not None:
        all_rows = list(cached_rows)
        # Plots can't be regenerated without re-running the sweep (they need
        # per-(pos, layer) score grids that aren't reconstructed here). The
        # PNGs from the original sweep remain on disk.
    else:
        print("[select_direction] computing baseline refusal scores")
        baseline_harmful_scores, _ = _score_dataset(
            model, harmful_val, None, refusal_toks
        )
        baseline_harmful = torch.tensor(baseline_harmful_scores)
        baseline_harmless_scores, baseline_harmless_logits = _score_dataset(
            model, harmless_val, None, refusal_toks
        )
        baseline_harmless = torch.tensor(baseline_harmless_scores)
        all_rows = _run_sweep(
            model,
            candidates,
            harmful_val,
            harmless_val,
            baseline_harmful,
            baseline_harmless,
            baseline_harmless_logits,
            pos_labels,
            config,
            out_dir,
        )
        _save_evals_with_meta(out_dir, all_rows, signature)

    filtered = [
        row
        for row in all_rows
        if not _filter(
            row["refusal_score"],
            row["induce_score"],
            row["kl_score"],
            row["layer"],
            n_layers,
            config,
        )
    ]
    filtered.sort(key=lambda r: r["refusal_score"])

    with (out_dir / "direction_evaluations_filtered.json").open("w") as f:
        json.dump(filtered, f, indent=2)

    if not filtered:
        raise RuntimeError(
            "All candidate directions were filtered out — relax thresholds "
            "or widen `intermediates`."
        )

    best = filtered[0]
    intermediate = best["intermediate"]
    n_pos = candidates[intermediate].shape[0]
    pos_idx = best["position"] + n_pos
    direction = candidates[intermediate][pos_idx, best["layer"]].to(torch.float32)

    metadata = {
        "intermediate": intermediate,
        "position": best["position"],
        "layer": best["layer"],
        "refusal_score": best["refusal_score"],
        "induce_score": best["induce_score"],
        "kl_score": best["kl_score"],
        "eval_signature": signature,
    }
    with (config.output_dir / "direction_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    torch.save(direction, config.output_dir / "direction.pt")

    return {
        **{k: v for k, v in metadata.items() if k != "eval_signature"},
        "direction": direction,
    }
