"""Steering operations for SAE features and raw direction vectors.

Four steering modes are supported, all applied as broadcasts across every
token position of a hidden state tensor ``h`` of shape ``(batch, seq, d_in)``
along a direction ``v`` of shape ``(d_in,)``:

- ``ADDITIVE``       : ``h' = h + strength * v``
- ``ORTHOGONAL``     : ``h' = h + (strength - 1) * ((h @ v) / (v @ v)) * v``
- ``ABLATION``       : ``h' = h + (strength - 1) * (h @ v) * v``  (v unit-norm)
- ``PROJECTION_CAP`` : ``proj = h @ v``,
                       ``h' = h + (clip(proj, cap_min, cap_max) - proj) * v``
                       (v unit-norm)

The rank-1 form keeps everything dtype/device-friendly and avoids ever
materialising a ``(d_in, d_in)`` projection matrix.
"""

from dataclasses import dataclass
from enum import Enum

import torch

from interpret.sae.sae_config import HookType
from interpret.sae.sae_model import SAEBase

_ZERO_NORM_TOL = 1e-8


class SteeringMode(Enum):
    """How a steering direction modifies the hidden state."""

    ADDITIVE = "additive"
    ORTHOGONAL = "orthogonal"
    ABLATION = "ablation"
    PROJECTION_CAP = "projection_cap"


@dataclass
class SteeringOp:
    """User-facing specification for a single steering operation.

    Exactly one of ``feature_index`` or ``vector`` must be set. For
    ``PROJECTION_CAP``, ``strength`` is ignored and at least one of
    ``cap_min`` / ``cap_max`` must be provided.

    Attributes:
        layer_index: Decoder layer to apply the op to.
        mode: Which steering formula to use.
        strength: Coefficient for additive / orthogonal / ablation modes.
        cap_min: Lower bound on ``h @ v`` for ``PROJECTION_CAP``.
        cap_max: Upper bound on ``h @ v`` for ``PROJECTION_CAP``.
        feature_index: Row index into ``sae.w_dec`` to use as the direction.
        vector: Raw direction vector of shape ``(d_in,)``.
        sae_key: Which registered SAE to resolve ``feature_index`` against.
                 Defaults to ``(layer_index, hook_type)``, i.e. the SAE at
                 the same site this op targets.
        normalise: L2-normalise the direction before use. ``ABLATION`` and
                   ``PROJECTION_CAP`` always normalise regardless of this flag.
        hook_type: Which site within the decoder layer to apply the op to.
                   ``RESID_POST`` (default) hooks the full layer output;
                   ``ATTN_OUT`` and ``MLP_OUT`` hook the attention / MLP
                   sub-modules respectively. Three ops on the same layer with
                   different ``hook_type`` values give the three-site
                   ablation pattern of Arditi et al.
    """

    layer_index: int
    mode: SteeringMode
    strength: float = 1.0
    cap_min: float | None = None
    cap_max: float | None = None
    feature_index: int | None = None
    vector: torch.Tensor | None = None
    sae_key: tuple[int, HookType] | None = None
    normalise: bool = False
    hook_type: HookType = HookType.RESID_POST

    def __post_init__(self) -> None:
        has_idx = self.feature_index is not None
        has_vec = self.vector is not None
        if has_idx == has_vec:
            raise ValueError(
                "SteeringOp requires exactly one of feature_index or vector"
            )
        if self.mode is SteeringMode.PROJECTION_CAP:
            if self.cap_min is None and self.cap_max is None:
                raise ValueError(
                    "PROJECTION_CAP requires at least one of cap_min / cap_max"
                )


@dataclass
class ResolvedSteeringOp:
    """A steering op with its direction materialised on the right device/dtype."""

    layer_index: int
    mode: SteeringMode
    strength: float
    cap_min: float | None
    cap_max: float | None
    v: torch.Tensor
    v_dot_v: float
    hook_type: HookType = HookType.RESID_POST


def resolve_op(
    op: SteeringOp,
    sae: SAEBase | None,
    device: torch.device,
    dtype: torch.dtype,
) -> ResolvedSteeringOp:
    """Materialise a SteeringOp into a ResolvedSteeringOp on the target layer.

    Validates the direction (finite, non-zero norm), resolves a feature
    index against the supplied SAE if needed, applies L2 normalisation
    when required, and casts to the target device + dtype.
    """
    if op.feature_index is not None:
        if sae is None:
            raise ValueError(
                f"SteeringOp at layer {op.layer_index} uses feature_index="
                f"{op.feature_index} but no SAE is registered at "
                f"sae_key={op.sae_key or (op.layer_index, HookType.RESID_POST)}. "
                "Call HookManager.add_sae(...) first or pass a raw `vector`."
            )
        if not (0 <= op.feature_index < sae.d_sae):
            raise ValueError(
                f"feature_index {op.feature_index} out of range "
                f"[0, {sae.d_sae})"
            )
        v = sae.w_dec[op.feature_index].detach().clone()
    else:
        assert op.vector is not None
        v = op.vector.detach().clone()

    if v.ndim != 1:
        raise ValueError(f"Steering vector must be 1-D, got shape {tuple(v.shape)}")
    if not torch.isfinite(v).all():
        raise ValueError("Steering vector contains non-finite values")

    # Compute norm in fp32 for stability, regardless of source dtype.
    v_fp32 = v.to(torch.float32)
    norm = float(torch.linalg.vector_norm(v_fp32).item())
    if norm < _ZERO_NORM_TOL:
        raise ValueError(f"Steering vector has near-zero norm ({norm:.2e})")

    force_normalise = op.mode in (SteeringMode.ABLATION, SteeringMode.PROJECTION_CAP)
    if op.normalise or force_normalise:
        v_fp32 = v_fp32 / norm
        v_dot_v = 1.0
    else:
        v_dot_v = norm * norm

    v_final = v_fp32.to(device=device, dtype=dtype)

    return ResolvedSteeringOp(
        layer_index=op.layer_index,
        mode=op.mode,
        strength=op.strength,
        cap_min=op.cap_min,
        cap_max=op.cap_max,
        v=v_final,
        v_dot_v=v_dot_v,
        hook_type=op.hook_type,
    )


def apply_steering(
    hidden_states: torch.Tensor,
    ops: list[ResolvedSteeringOp],
    strength_multiplier: float = 1.0,
) -> torch.Tensor:
    """Apply a list of resolved steering ops to a hidden state tensor.

    Returns a new tensor; ``hidden_states`` is not modified in place. Ops
    are applied in list order. ``strength_multiplier`` is ignored for
    ``PROJECTION_CAP`` (the bounds are absolute).

    Args:
        hidden_states: Tensor of shape ``(batch, seq, d_in)`` (or any shape
                       whose last dim is ``d_in``).
        ops: Resolved ops, all targeting this tensor's last-dim space.
        strength_multiplier: Global scalar multiplied into ``op.strength``
                             for additive / orthogonal / ablation modes.

    Returns:
        New hidden state tensor with all ops applied.
    """
    h = hidden_states
    for op in ops:
        v = op.v
        if op.mode is SteeringMode.ADDITIVE:
            coeff = strength_multiplier * op.strength
            h = h + coeff * v
        elif op.mode is SteeringMode.ORTHOGONAL:
            coeff = strength_multiplier * op.strength
            proj = (h @ v) / op.v_dot_v
            h = h + (coeff - 1.0) * proj.unsqueeze(-1) * v
        elif op.mode is SteeringMode.ABLATION:
            coeff = strength_multiplier * op.strength
            proj = h @ v
            h = h + (coeff - 1.0) * proj.unsqueeze(-1) * v
        elif op.mode is SteeringMode.PROJECTION_CAP:
            proj = h @ v
            clamped = proj
            if op.cap_min is not None:
                clamped = torch.clamp(clamped, min=op.cap_min)
            if op.cap_max is not None:
                clamped = torch.clamp(clamped, max=op.cap_max)
            h = h + (clamped - proj).unsqueeze(-1) * v
        else:
            raise ValueError(f"Unknown steering mode: {op.mode}")
    return h
