"""Three-metric sweep + best-direction selection.

Adapted from `references/refusal_direction/pipeline/submodules/select_direction.py`.

For every candidate ``(intermediate, position, layer)``:

* **bypass score**  — refusal score on harmful_val under ablation at every
  ``(layer, hook_type)`` for ``hook_type in {RESID_POST, ATTN_OUT, MLP_OUT}``;
* **induce score** — refusal score on harmless_val under additive steering at
  the source layer (RESID_POST);
* **KL on harmless** — KL between baseline last-position logits and ablated
  last-position logits over harmless_val.

Filtering follows the paper: drop the top ``prune_layer_pct`` of layers, drop
KL > kl_threshold, drop induce score < induce_refusal_threshold. The remaining
direction with the lowest refusal score under ablation wins.
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from interpret.experiments.refusal_directions.config import RefusalConfig
from interpret.experiments.refusal_directions.tokens import format_chat
from interpret.sae import HookManager, HookType, SteeringMode, SteeringOp

_REFUSAL_EPSILON = 1e-8
_KL_EPSILON = 1e-6
_ZERO_NORM_TOL = 1e-8


def _last_position_logits(wrapper, prompt: str) -> torch.Tensor:
    """Forward pass on `prompt`; return last-position logits (CPU, fp32).

    Captures the post-final-norm hidden state via `cache_activations` and
    applies the model's tied embedding (with optional softcap) — the same
    logit pipeline that ``Sampler`` uses inside ``model.generate``. Any
    `HookManager.session(...)` opened by the caller is in effect during this
    forward pass; this function does not manage hooks itself, so callers can
    keep a steering session open across an entire batch.
    """
    # `cache_activations` only emits a "prefill" key when `_prefill_cache`
    # is non-empty (gemma_pytorch.py:236 — empty dict is falsy). We don't
    # use the per-layer cache here, but we need at least one layer in the
    # set so the prefill snapshot is taken and `final_norm` makes it through.
    with wrapper.cache_activations(
        layers={0}, intermediates={"final_norm"}, prefill=True
    ) as get_cache:
        wrapper.reset_prefill_cache()
        wrapper.generate_from_template(
            format_chat(wrapper, prompt), output_len=1
        )
        cache = get_cache()

    final_norm = cache["prefill"]["final_norm"]  # (1, seq_len, d_model), CPU
    embed = wrapper.model.text_token_embedder.weight  # (vocab, d_model)
    last = final_norm[0, -1, :].to(embed.device, embed.dtype)
    logits = last @ embed.T
    cap = getattr(wrapper.config, "final_logit_softcapping", None)
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits.detach().to("cpu", torch.float32)


@contextlib.contextmanager
def _maybe_session(manager: HookManager | None, layers):
    """Open a HookManager session if one was provided, else a no-op."""
    if manager is None:
        yield
        return
    with manager.session(layers):
        yield


def _score_dataset(
    wrapper,
    prompts: list[str],
    manager: HookManager | None,
    refusal_toks: tuple[int, ...],
) -> tuple[list[float], list[torch.Tensor]]:
    """Run forward over `prompts` once, returning (refusal_scores, last_logits)."""
    layers = wrapper.model.model.layers
    scores: list[float] = []
    logits_list: list[torch.Tensor] = []
    with _maybe_session(manager, layers):
        for prompt in prompts:
            logits = _last_position_logits(wrapper, prompt)
            scores.append(_refusal_score(logits, refusal_toks))
            logits_list.append(logits)
    return scores, logits_list


def _refusal_score(logits: torch.Tensor, refusal_toks: tuple[int, ...]) -> float:
    """Reference score: log P(refusal_tok) - log(1 - P(refusal_tok))."""
    logits = logits.to(torch.float64)
    probs = torch.softmax(logits, dim=-1)
    refusal_p = probs[list(refusal_toks)].sum().item()
    nonref_p = max(1.0 - refusal_p, _REFUSAL_EPSILON)
    return math.log(refusal_p + _REFUSAL_EPSILON) - math.log(nonref_p)


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) at a single position."""
    p = torch.softmax(p_logits.to(torch.float64), dim=-1)
    q = torch.softmax(q_logits.to(torch.float64), dim=-1)
    return torch.sum(
        p * (torch.log(p + _KL_EPSILON) - torch.log(q + _KL_EPSILON))
    ).item()


def _ablation_ops(direction: torch.Tensor, n_layers: int) -> list[SteeringOp]:
    """Three-site ablation ops: every layer × {RESID_POST, ATTN_OUT, MLP_OUT}.

    ``strength=0.0`` triggers full projection ablation — the formula
    ``h + (strength - 1) * (h @ v) * v`` reduces to ``h - (h @ v) * v`` when
    strength is zero (see ``interpret/sae/steering.py``).

    Boundary divergence vs. the reference: the reference attaches a
    forward-pre-hook on each decoder layer (operating on the layer *input* —
    same residual point as the previous layer's output), whereas this project
    attaches a forward-hook on the layer *output*. The two coincide on every
    inter-layer boundary; they differ at the extremes:
    - layer 0 *input* is ablated by the reference but not by us;
    - the residual *after* the final layer (input to the final norm) is
      ablated by us but not by the reference.
    Net effect on Gemma-3-4b is a 1-of-34 site offset.
    """
    ops: list[SteeringOp] = []
    for layer in range(n_layers):
        for hook_type in (HookType.RESID_POST, HookType.ATTN_OUT, HookType.MLP_OUT):
            ops.append(
                SteeringOp(
                    layer_index=layer,
                    mode=SteeringMode.ABLATION,
                    vector=direction.detach().clone(),
                    strength=0.0,
                    hook_type=hook_type,
                )
            )
    return ops


def _additive_op(
    direction: torch.Tensor, layer: int, coeff: float
) -> SteeringOp:
    return SteeringOp(
        layer_index=layer,
        mode=SteeringMode.ADDITIVE,
        vector=direction.detach().clone(),
        strength=coeff,
        hook_type=HookType.RESID_POST,
    )


def _make_manager(ops: list[SteeringOp]) -> HookManager:
    manager = HookManager()
    if ops:
        manager.add_steering(ops)
    return manager


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
    wrapper,
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

    direction_path = config.output_dir / "direction.pt"
    metadata_path = config.output_dir / "direction_metadata.json"
    if direction_path.exists() and metadata_path.exists():
        print(f"[select_direction] reusing cached {direction_path}")
        with metadata_path.open() as f:
            metadata = json.load(f)
        return {**metadata, "direction": torch.load(direction_path, map_location="cpu")}

    n_layers = config.n_layers
    refusal_toks = config.refusal_token_ids

    print("[select_direction] computing baseline refusal scores")
    baseline_harmful_scores, _ = _score_dataset(
        wrapper, harmful_val, None, refusal_toks
    )
    baseline_harmful = torch.tensor(baseline_harmful_scores)
    baseline_harmless_scores, baseline_harmless_logits = _score_dataset(
        wrapper, harmless_val, None, refusal_toks
    )
    baseline_harmless = torch.tensor(baseline_harmless_scores)

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
                desc=f"sweep {intermediate} pos={source_pos - n_pos}",
            ):
                direction = candidate_tensor[source_pos, source_layer].to(
                    torch.float32
                )
                if direction.norm().item() < _ZERO_NORM_TOL:
                    # Layer 0's pre_attn (and any other position whose
                    # mean diff happens to vanish — e.g. when every prompt
                    # has identical token IDs at that EOI position before
                    # any attention has run) yields a zero direction. Score
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

                # Bypass: ablate at every layer × hook_type, score on harmful_val.
                ablation_mgr = _make_manager(_ablation_ops(direction, n_layers))
                ablated_scores, _ = _score_dataset(
                    wrapper, harmful_val, ablation_mgr, refusal_toks
                )
                ablation_refusal[source_pos, source_layer] = float(
                    sum(ablated_scores) / max(len(ablated_scores), 1)
                )

                # KL on harmless under the same ablation (re-uses the manager).
                _, ablated_harmless_logits = _score_dataset(
                    wrapper, harmless_val, ablation_mgr, refusal_toks
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
                    wrapper, harmless_val, induce_mgr, refusal_toks
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
            f"Ablation on harmful — {intermediate}",
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
            f"KL divergence under ablation — {intermediate}",
            out_dir / f"kl_div_scores_{intermediate}.png",
        )

    with (out_dir / "direction_evaluations.json").open("w") as f:
        json.dump(all_rows, f, indent=2)

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
    }
    with (config.output_dir / "direction_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    torch.save(direction, config.output_dir / "direction.pt")

    return {
        **metadata,
        "direction": direction,
    }
