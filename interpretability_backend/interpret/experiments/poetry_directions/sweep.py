"""Layer × coefficient additive-bypass sweep with KL coherence filter.

For every candidate ``(intermediate, source_position, source_layer, coefficient)``:

* **bypass score** — refusal score on harmful_val under one ADDITIVE op at the
  source layer with the given coefficient (the same primitive validated on
  Gemma-3 by ``refusal_steer_tester.ipynb``).
* **KL on harmless** — KL between baseline last-position logits and the same
  intervention's last-position logits over harmless_val (coherence guard;
  catches coefficients that collapse the residual stream).

Filtering mirrors ``interpret.experiments.refusal_directions.select_direction._filter`` but
without the ``induce`` term — there's no separate inducing intervention here;
the bypass refusal score is the headline criterion. The remaining direction
with the lowest bypass score wins.
"""

from __future__ import annotations

import csv as _csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from interpret.experiments.poetry_directions.config import PoetryConfig
from interpret.experiments.refusal_directions.select_direction import (
    _additive_op,
    _kl_div,
    _make_manager,
    _score_dataset,
)
from interpret.utils.results_io import append_csv


_ZERO_NORM_TOL = 1e-8


def _parse_float(s: str) -> float:
    """Parse a CSV float cell, treating empty / 'nan' / 'NaN' as NaN."""
    if s is None or s == "" or s.lower() == "nan":
        return float("nan")
    return float(s)


def _load_existing_rows(live_csv: Path) -> tuple[list[dict], dict[tuple, dict]]:
    """Load any rows previously written to `live_csv`.

    Returns a tuple of (rows_list, done_index) where ``done_index`` maps
    ``(intermediate, position, layer, coefficient)`` → row, used both for
    skipping cells in the loop and for repopulating the in-memory grids
    from cached values.
    """
    if not live_csv.exists():
        return [], {}
    rows: list[dict] = []
    index: dict[tuple, dict] = {}
    with live_csv.open() as f:
        reader = _csv.DictReader(f)
        for raw in reader:
            row = {
                "intermediate": raw["intermediate"],
                "position": int(raw["position"]),
                "layer": int(raw["layer"]),
                "coefficient": float(raw["coefficient"]),
                "refusal_score": _parse_float(raw.get("refusal_score", "")),
                "kl_score": _parse_float(raw.get("kl_score", "")),
                "skipped": raw.get("skipped", "") or "",
            }
            rows.append(row)
            index[
                (row["intermediate"], row["position"], row["layer"], row["coefficient"])
            ] = row
    return rows, index


def _baseline_cache_path(cfg: PoetryConfig) -> Path:
    """Path under ``output_dir_root/_baselines/`` keyed by params that affect baselines.

    Shared across all experiments that consume the same model + val splits.
    """
    model_alias = cfg.model_name.replace("/", "_")
    refusal_str = "-".join(str(t) for t in cfg.refusal_token_ids)
    fname = (
        f"{model_alias}_nval{cfg.n_val}_seed{cfg.seed}_refusal{refusal_str}.pt"
    )
    return cfg.output_dir_root / "_baselines" / fname


def _load_or_compute_baselines(
    wrapper,
    harmful_val: list[str],
    harmless_val: list[str],
    refusal_toks: tuple[int, ...],
    cache_path: Path,
) -> tuple[list[float], list[float], list[torch.Tensor]]:
    """Load cached baseline (refusal scores + harmless last-position logits) or compute + save.

    Cache invalidates automatically when ``len(harmful_val) / len(harmless_val)``
    differ from the cached lengths (covers a stale cache after `n_val` changes
    or seed-driven sample drift).
    """
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if (
            len(cached.get("harmful_scores", [])) == len(harmful_val)
            and len(cached.get("harmless_scores", [])) == len(harmless_val)
            and len(cached.get("harmless_logits", [])) == len(harmless_val)
        ):
            print(f"[sweep] reusing cached baseline at {cache_path}")
            return (
                list(cached["harmful_scores"]),
                list(cached["harmless_scores"]),
                list(cached["harmless_logits"]),
            )
        print(f"[sweep] baseline cache mismatch at {cache_path}, recomputing")

    print("[sweep] computing baseline last-position logits (will cache)")
    harmful_scores, _ = _score_dataset(wrapper, harmful_val, None, refusal_toks)
    harmless_scores, harmless_logits = _score_dataset(
        wrapper, harmless_val, None, refusal_toks
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "harmful_scores": harmful_scores,
            "harmless_scores": harmless_scores,
            "harmless_logits": harmless_logits,
        },
        cache_path,
    )
    return harmful_scores, harmless_scores, harmless_logits


def _filter(
    refusal_score: float,
    kl_score: float,
    layer: int,
    n_layers: int,
    cfg: PoetryConfig,
) -> bool:
    """Return True to discard the candidate."""
    if any(math.isnan(s) for s in (refusal_score, kl_score)):
        return True
    if layer >= int(n_layers * (1.0 - cfg.prune_layer_pct)):
        return True
    if kl_score > cfg.kl_threshold:
        return True
    return False


def _plot_grid(
    grid: torch.Tensor,
    coeffs: tuple[float, ...],
    baseline: float | None,
    title: str,
    out_path: Path,
) -> None:
    """Heatmap of `grid` (layers × coeffs). Annotates baseline if provided."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid.cpu().numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(coeffs)))
    ax.set_xticklabels([f"{c:+.2f}" for c in coeffs], rotation=45)
    ax.set_xlabel("coefficient")
    ax.set_ylabel("source layer")
    ax.set_title(title + (f"  (baseline={baseline:+.3f})" if baseline is not None else ""))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_position_curves(
    scores: torch.Tensor,
    pos_labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Per-position curves over layers (averaged over coefficients)."""
    n_pos, n_layers, _n_coeff = scores.shape
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(-n_pos, 0):
        # Take the BEST coefficient per (pos, layer).
        per_layer = scores[i].min(dim=-1).values
        ax.plot(range(n_layers), per_layer.cpu().numpy(), label=f"{i}: {pos_labels[i]!r}")
    ax.set_xlabel("source layer")
    ax.set_ylabel("score (best coefficient)")
    ax.set_title(title)
    ax.legend(title="position", loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def sweep_layers_coeffs(
    wrapper,
    candidates: dict[str, torch.Tensor],
    harmful_val: list[str],
    harmless_val: list[str],
    pos_labels: list[str],
    cfg: PoetryConfig,
) -> dict:
    """Run the (intermediate, position, layer, coefficient) sweep + select best.

    Returns a dict with the chosen candidate and writes:
      - ``cfg.sweep_dir/direction_evaluations.json`` (every candidate)
      - ``cfg.sweep_dir/direction_evaluations_filtered.json`` (after filter)
      - ``cfg.sweep_dir/refusal_scores_<intermediate>.png``
      - ``cfg.sweep_dir/kl_scores_<intermediate>.png``
      - ``cfg.sweep_dir/bypass_grid_<intermediate>.png``
      - ``cfg.output_dir/direction.pt`` (the chosen vector)
      - ``cfg.output_dir/direction_metadata.json``
    Idempotent: skips the sweep entirely when ``direction.pt`` and the
    metadata file are already present.
    """
    sweep_dir = cfg.sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    direction_path = cfg.output_dir / "direction.pt"
    metadata_path = cfg.output_dir / "direction_metadata.json"
    if direction_path.exists() and metadata_path.exists():
        print(f"[sweep:{cfg.name}] reusing cached {direction_path}")
        with metadata_path.open() as f:
            metadata = json.load(f)
        return {**metadata, "direction": torch.load(direction_path, map_location="cpu")}

    n_layers = cfg.n_layers
    coeffs = cfg.coefficients
    refusal_toks = cfg.refusal_token_ids

    # Resume: load any previously-written rows so already-computed cells are
    # skipped on restart. all_rows is seeded with the cached rows; the loop
    # below appends only newly-computed ones to both all_rows and the CSV.
    live_csv = sweep_dir / "direction_evaluations.csv"
    cached_rows, done_index = _load_existing_rows(live_csv)
    all_rows: list[dict] = list(cached_rows)
    if cached_rows:
        print(
            f"[sweep:{cfg.name}] resuming with {len(cached_rows)} cells "
            f"already in {live_csv}"
        )

    baseline_harmful_scores, baseline_harmless_scores, baseline_harmless_logits = (
        _load_or_compute_baselines(
            wrapper,
            harmful_val,
            harmless_val,
            refusal_toks,
            _baseline_cache_path(cfg),
        )
    )
    baseline_harmful_mean = float(
        sum(baseline_harmful_scores) / max(len(baseline_harmful_scores), 1)
    )
    baseline_harmless_mean = float(
        sum(baseline_harmless_scores) / max(len(baseline_harmless_scores), 1)
    )

    for intermediate, candidate_tensor in candidates.items():
        n_pos, layer_count, _ = candidate_tensor.shape
        assert layer_count == n_layers, (
            f"candidate shape mismatch: {candidate_tensor.shape} vs n_layers={n_layers}"
        )

        refusal_grid = torch.full((n_pos, n_layers, len(coeffs)), float("nan"))
        kl_grid = torch.full((n_pos, n_layers, len(coeffs)), float("nan"))

        # Repopulate grids from cached rows so the plots cover the full
        # history of work (current sweep + any previously-computed cells).
        for cached in cached_rows:
            if cached["intermediate"] != intermediate:
                continue
            pos_idx = cached["position"] + n_pos
            if not 0 <= pos_idx < n_pos:
                continue
            l = cached["layer"]
            if not 0 <= l < n_layers:
                continue
            if cached["coefficient"] not in coeffs:
                continue
            c_idx = coeffs.index(cached["coefficient"])
            rs, ks = cached["refusal_score"], cached["kl_score"]
            if not math.isnan(rs):
                refusal_grid[pos_idx, l, c_idx] = rs
            if not math.isnan(ks):
                kl_grid[pos_idx, l, c_idx] = ks

        if cfg.positions is None:
            pos_indices = list(range(n_pos))
        else:
            pos_indices = [
                p + n_pos for p in cfg.positions if -n_pos <= p < 0
            ]
            if not pos_indices:
                raise ValueError(
                    f"cfg.positions={cfg.positions} produced no valid indices "
                    f"for tensor of n_pos={n_pos}"
                )

        for source_pos in pos_indices:
            pos_label = source_pos - n_pos
            for source_layer in tqdm(
                range(n_layers),
                desc=f"sweep {cfg.name} {intermediate} pos={pos_label}",
            ):
                def _record(row: dict) -> None:
                    all_rows.append(row)
                    csv_row = {
                        "intermediate": row["intermediate"],
                        "position": row["position"],
                        "layer": row["layer"],
                        "coefficient": row["coefficient"],
                        "refusal_score": row["refusal_score"],
                        "kl_score": row["kl_score"],
                        "skipped": row.get("skipped", ""),
                    }
                    append_csv(live_csv, csv_row)

                direction = candidate_tensor[source_pos, source_layer].to(torch.float32)
                if direction.norm().item() < _ZERO_NORM_TOL:
                    for c_idx in range(len(coeffs)):
                        if (intermediate, pos_label, source_layer, coeffs[c_idx]) in done_index:
                            continue
                        _record(
                            {
                                "intermediate": intermediate,
                                "position": pos_label,
                                "layer": source_layer,
                                "coefficient": coeffs[c_idx],
                                "refusal_score": float("nan"),
                                "kl_score": float("nan"),
                                "skipped": "zero_norm",
                            }
                        )
                    continue

                v_norm = direction.norm().item()
                for c_idx, coeff in enumerate(coeffs):
                    if (intermediate, pos_label, source_layer, coeff) in done_index:
                        # Already computed in a previous run; grid was populated
                        # from the cache above.
                        continue
                    if abs(coeff) * v_norm > cfg.magnitude_cap:
                        _record(
                            {
                                "intermediate": intermediate,
                                "position": pos_label,
                                "layer": source_layer,
                                "coefficient": coeff,
                                "refusal_score": float("nan"),
                                "kl_score": float("nan"),
                                "skipped": "magnitude_cap",
                            }
                        )
                        continue
                    manager = _make_manager(
                        [_additive_op(direction, source_layer, coeff=coeff)]
                    )
                    bypass_scores, _ = _score_dataset(
                        wrapper, harmful_val, manager, refusal_toks
                    )
                    refusal_mean = float(
                        sum(bypass_scores) / max(len(bypass_scores), 1)
                    )
                    refusal_grid[source_pos, source_layer, c_idx] = refusal_mean

                    _, intervened_harmless_logits = _score_dataset(
                        wrapper, harmless_val, manager, refusal_toks
                    )
                    kls = [
                        _kl_div(b, a)
                        for b, a in zip(
                            baseline_harmless_logits, intervened_harmless_logits
                        )
                    ]
                    kl_mean = float(sum(kls) / max(len(kls), 1))
                    kl_grid[source_pos, source_layer, c_idx] = kl_mean

                    _record(
                        {
                            "intermediate": intermediate,
                            "position": pos_label,
                            "layer": source_layer,
                            "coefficient": coeff,
                            "refusal_score": refusal_mean,
                            "kl_score": kl_mean,
                        }
                    )

        # Plots: pick a "best coefficient per (pos, layer)" view for the curves,
        # and per-coefficient × per-layer heatmap averaged over positions.
        _plot_position_curves(
            refusal_grid,
            pos_labels,
            f"refusal (best coefficient) — {cfg.name} / {intermediate}\n"
            f"baseline harmful = {baseline_harmful_mean:+.3f}",
            sweep_dir / f"refusal_scores_{intermediate}.png",
        )
        _plot_position_curves(
            kl_grid,
            pos_labels,
            f"KL (best coefficient) — {cfg.name} / {intermediate}",
            sweep_dir / f"kl_scores_{intermediate}.png",
        )
        avg_grid = torch.nanmean(refusal_grid, dim=0)  # (n_layers, n_coeff)
        _plot_grid(
            avg_grid,
            coeffs,
            baseline_harmful_mean,
            f"refusal score — averaged over positions — {cfg.name} / {intermediate}",
            sweep_dir / f"bypass_grid_{intermediate}.png",
        )

    with (sweep_dir / "direction_evaluations.json").open("w") as f:
        json.dump(all_rows, f, indent=2)

    filtered = [
        row
        for row in all_rows
        if not _filter(
            row["refusal_score"], row["kl_score"], row["layer"], n_layers, cfg
        )
    ]
    filtered.sort(key=lambda r: r["refusal_score"])
    with (sweep_dir / "direction_evaluations_filtered.json").open("w") as f:
        json.dump(filtered, f, indent=2)

    if not filtered:
        raise RuntimeError(
            f"All candidates filtered out for experiment {cfg.name!r}. "
            "Inspect direction_evaluations.json and consider raising "
            "kl_threshold or widening the coefficients grid."
        )

    best = filtered[0]
    intermediate = best["intermediate"]
    n_pos = candidates[intermediate].shape[0]
    pos_idx = best["position"] + n_pos
    direction = candidates[intermediate][pos_idx, best["layer"]].to(torch.float32)

    metadata = {
        "name": cfg.name,
        "intermediate": intermediate,
        "position": best["position"],
        "layer": best["layer"],
        "coefficient": best["coefficient"],
        "refusal_score": best["refusal_score"],
        "kl_score": best["kl_score"],
        "baseline_harmful_refusal_score": baseline_harmful_mean,
        "baseline_harmless_refusal_score": baseline_harmless_mean,
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    torch.save(direction, direction_path)

    return {**metadata, "direction": direction}
