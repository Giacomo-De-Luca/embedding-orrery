"""Steering-op constructors shared by the refusal and poetry pipelines.

Moved verbatim from ``refusal_directions/select_direction.py`` so both
experiments (and any future model) build interventions through one place.
These are model-agnostic: they emit ``SteeringOp`` objects that
``HookManager`` attaches to whatever decoder ``ModuleList`` it is given.
"""

from __future__ import annotations

import torch

from interpret.sae import HookManager, HookType, SteeringMode, SteeringOp


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
    Net effect is a 1-of-n_layers site offset.
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


def _additive_op(direction: torch.Tensor, layer: int, coeff: float) -> SteeringOp:
    """Single additive op at ``layer`` (RESID_POST) with the given coefficient."""
    return SteeringOp(
        layer_index=layer,
        mode=SteeringMode.ADDITIVE,
        vector=direction.detach().clone(),
        strength=coeff,
        hook_type=HookType.RESID_POST,
    )


def _make_manager(ops: list[SteeringOp]) -> HookManager:
    """Build a HookManager pre-loaded with ``ops`` (empty = identity manager)."""
    manager = HookManager()
    if ops:
        manager.add_steering(ops)
    return manager


def _bypass_ops(
    direction: torch.Tensor,
    n_layers: int,
    source_layer: int,
    bypass_mode: str,
    actadd_coeff: float = -1.0,
) -> list[SteeringOp]:
    """Build the bypass-intervention ops for the selection sweep.

    Dispatches on ``bypass_mode``:

    - ``"ablation"`` — three-site projection ablation at every layer ×
      {RESID_POST, ATTN_OUT, MLP_OUT}. Arditi's original recipe; works for
      Gemma 1/2, Llama, classic Qwen.
    - ``"actadd"`` — single additive op at ``source_layer`` with
      ``actadd_coeff`` (typically -1, "subtract refusal"). Used when full
      ablation collapses the residual stream: this happens on Gemma-3
      (huge post-norm activation norms) and also on Qwen3 (where every
      layer × three-site ablation drives logits to ``+inf``).

    ``source_layer`` is ignored in ablation mode (which hits every layer).
    """
    if bypass_mode == "ablation":
        return _ablation_ops(direction, n_layers)
    if bypass_mode == "actadd":
        return [_additive_op(direction, source_layer, coeff=actadd_coeff)]
    raise ValueError(
        f"Unknown bypass_mode: {bypass_mode!r}. Valid: 'ablation', 'actadd'."
    )
