"""Canonical residual-stream capture sites, decoupled from any one backend.

The refusal / poetry direction pipelines extract candidate directions at a
handful of points inside each decoder layer. Historically those points were
named with Gemma's internal-cache strings (``pre_attn`` / ``post_attn`` /
``mlp_out`` / ``post_mlp``). ``CaptureSite`` keeps those strings as the
canonical identifiers — so config fields, ``mean_diffs_<site>.pt`` artifact
names, and JSON/CSV columns are unchanged — while giving each a semantic name
and a per-backend resolution.

Gemma resolves a site directly to its fork-cache intermediate name
(``CaptureSite.value``). Qwen has no ``pre_attn`` capture point and a
different naming, so each site maps to a ``(HookType, layer_offset)`` pair:

    RESID_PRE[L]  == RESID_POST[L-1]   (block input == previous block output)
    POST_ATTN[L]  == POST_ATTN[L]
    MLP_OUT[L]    == MLP_OUT[L]
    RESID_POST[L] == RESID_POST[L]

``RESID_PRE`` at layer 0 (the embedding output, before any attention) has no
``RESID_POST[-1]`` source; the adapter leaves it as zeros. This matches
Gemma, where ``pre_attn`` at layer 0 over the constant end-of-instruction
positions is identically zero (the suffix tokens are the same across prompts,
so their pre-attention mean-diff vanishes) and is discarded by the zero-norm
filter downstream.
"""

from __future__ import annotations

from enum import Enum

from interpret.sae.sae_config import HookType


class CaptureSite(Enum):
    """A residual-stream point to read a candidate direction from.

    The enum *value* is the canonical string used in configs and artifacts
    (kept identical to Gemma's fork-cache intermediate names for backward
    compatibility).
    """

    RESID_PRE = "pre_attn"  # residual stream at layer entry (block input)
    POST_ATTN = "post_attn"  # residual after attn-residual-add, before MLP norm
    MLP_OUT = "mlp_out"  # raw MLP block output (pre-residual-add)
    RESID_POST = "post_mlp"  # residual stream at layer exit (block output)

    @classmethod
    def from_name(cls, name: str) -> CaptureSite:
        """Resolve a canonical string (e.g. ``"pre_attn"``) to a CaptureSite."""
        for site in cls:
            if site.value == name:
                return site
        valid = sorted(s.value for s in cls)
        raise ValueError(f"unknown capture site {name!r}. Valid: {valid}")


# Qwen resolution: site -> (HookType to capture, layer offset to read from).
# RESID_PRE reads the *previous* layer's RESID_POST; layer 0 has no source and
# is left as zeros by the adapter (see module docstring).
QWEN_SITE_MAP: dict[CaptureSite, tuple[HookType, int]] = {
    CaptureSite.RESID_PRE: (HookType.RESID_POST, -1),
    CaptureSite.POST_ATTN: (HookType.POST_ATTN, 0),
    CaptureSite.MLP_OUT: (HookType.MLP_OUT, 0),
    CaptureSite.RESID_POST: (HookType.RESID_POST, 0),
}
