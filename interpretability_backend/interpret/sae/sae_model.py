"""Sparse Autoencoder modules.

Two pretrained-SAE families are supported, both exposing the same
``encode/decode/forward`` interface plus a ``w_dec`` of shape
``(d_sae, d_in)`` so that downstream code (HookManager, steering,
decoder-vector extraction) is family-agnostic:

- ``JumpReLUSAE`` — Gemma-scope architecture: input centring by
  ``b_dec``, then encoder + JumpReLU mask gated by a learned
  per-feature ``threshold``.
- ``TopKSAE`` — Qwen-scope architecture: encoder + hard top-k selection
  on the pre-activations (no centring, no ReLU on the kept values; raw
  top-k values are scattered into the sparse output, mirroring the
  Qwen-scope model card snippet).

The loader (``loading.load_sae``) is responsible for transposing weights
into the Gemma convention (``w_enc: (d_in, d_sae)``,
``w_dec: (d_sae, d_in)``) regardless of the on-disk layout.
"""

from pathlib import Path

import torch
from torch import nn


class SAEBase(nn.Module):
    """Common interface for pretrained SAE modules.

    Subclasses must define ``encode`` and ``decode``. ``forward`` returns
    ``(feature_acts, reconstruction)``. Subclasses must also expose
    ``w_dec`` with shape ``(d_sae, d_in)`` so that
    ``w_dec[feature_index]`` returns a ``(d_in,)`` direction vector
    suitable for steering.
    """

    d_in: int
    d_sae: int

    def __init__(self, d_in: int, d_sae: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_acts = self.encode(x)
        return feature_acts, self.decode(feature_acts)


class JumpReLUSAE(SAEBase):
    """JumpReLU SAE for inference on pretrained Gemma-scope weights.

    Architecture: input -> centre -> encode -> JumpReLU -> decode
    Weights are loaded from Gemma-scope safetensors files.
    """

    def __init__(self, d_in: int, d_sae: int) -> None:
        super().__init__(d_in, d_sae)
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations via JumpReLU."""
        pre_acts = (x - self.b_dec) @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold).to(pre_acts.dtype)
        return pre_acts * mask

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from feature activations."""
        return feature_acts @ self.w_dec + self.b_dec

    @classmethod
    def from_pretrained(
        cls,
        path: Path,
        d_in: int,
        d_sae: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "JumpReLUSAE":
        """Load from a Gemma-scope params.safetensors file."""
        from safetensors.torch import load_file

        state = load_file(str(path))
        sae = cls(d_in, d_sae)
        sae.w_enc.data = state["w_enc"].to(dtype=dtype)
        sae.w_dec.data = state["w_dec"].to(dtype=dtype)
        sae.b_enc.data = state["b_enc"].to(dtype=dtype)
        sae.b_dec.data = state["b_dec"].to(dtype=dtype)
        sae.threshold.data = state["threshold"].to(dtype=dtype)
        return sae.to(device)


class TopKSAE(SAEBase):
    """Hard TopK SAE for inference on pretrained Qwen-scope weights.

    Architecture: encode -> top-k selection -> scatter -> decode.
    No input centring (no ``(x - b_dec)``) and no ReLU on the kept
    activations — raw top-k values are passed through, matching the
    Qwen-scope model card snippet exactly.

    ``k`` is stored as a plain Python int (not an ``nn.Parameter``) since
    it is a fixed hyperparameter of the trained checkpoint, not a
    learned weight.
    """

    def __init__(self, d_in: int, d_sae: int, k: int) -> None:
        super().__init__(d_in, d_sae)
        if k <= 0 or k > d_sae:
            raise ValueError(f"k must be in (0, d_sae={d_sae}], got {k}")
        self.k = k
        self.w_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations via hard top-k."""
        pre_acts = x @ self.w_enc + self.b_enc
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        out = torch.zeros_like(pre_acts)
        return out.scatter_(-1, topk_idx, topk_vals)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from feature activations."""
        return feature_acts @ self.w_dec + self.b_dec
