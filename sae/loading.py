"""Download and load pretrained SAEs from HuggingFace.

Two families are supported with a single ``load_sae(config)`` entry
point that dispatches on the config dataclass type:

- ``GemmaScopeSAEConfig`` -> ``JumpReLUSAE`` from a Gemma-scope
  ``params.safetensors`` file.
- ``QwenScopeSAEConfig`` -> ``TopKSAE`` from a Qwen-scope
  ``layer{N}.sae.pt`` file (a plain ``torch.load`` dict).

Both loaders normalise weight orientation to the Gemma convention
(``w_enc: (d_in, d_sae)``, ``w_dec: (d_sae, d_in)``) so downstream code
(steering, decoder-vector extraction) is family-agnostic.
"""

from pathlib import Path

import torch

from interpret.sae.sae_config import (
    GemmaScopeSAEConfig,
    QwenScopeSAEConfig,
    WIDTH_TO_D_SAE,
)
from interpret.sae.sae_model import JumpReLUSAE, SAEBase, TopKSAE

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def download_sae_weights(
    repo_id: str,
    hook_type: str,
    layer_index: int,
    width: str,
    l0_size: str = "medium",
) -> Path:
    """Download Gemma-scope SAE weights from HuggingFace.

    Gemma-scope repo structure:
        {hook_type}/layer_{N}_width_{W}_l0_{size}/params.safetensors
    """
    from huggingface_hub import hf_hub_download

    folder = f"layer_{layer_index}_width_{width}_l0_{l0_size}"
    filename = f"{hook_type}/{folder}/params.safetensors"
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _load_gemma_scope_sae(config: GemmaScopeSAEConfig) -> JumpReLUSAE:
    dtype = DTYPE_MAP[config.dtype]
    path = download_sae_weights(
        config.repo_id,
        config.hook_type.value,
        config.layer_index,
        config.width,
        config.l0_size,
    )
    sae = JumpReLUSAE.from_pretrained(
        path, d_in=config.d_in, d_sae=WIDTH_TO_D_SAE[config.width],
        device=config.device, dtype=dtype,
    )
    sae.eval()
    return sae


def _load_qwen_scope_sae(config: QwenScopeSAEConfig) -> TopKSAE:
    """Download and load a Qwen-scope TopK SAE.

    The on-disk layout is ``W_enc: (d_sae, d_in)`` and ``W_dec: (d_in,
    d_sae)`` — transposed relative to the Gemma convention. We transpose
    on load so that ``x @ sae.w_enc`` and ``feature_acts @ sae.w_dec``
    work without any extra transposes downstream and so that
    ``sae.w_dec[feature_index]`` returns a ``(d_in,)`` direction vector
    suitable for steering.
    """
    from huggingface_hub import hf_hub_download

    dtype = DTYPE_MAP[config.dtype]
    path = Path(
        hf_hub_download(
            repo_id=config.repo_id,
            filename=config.weights_filename(),
        )
    )
    state = torch.load(path, map_location="cpu", weights_only=True)

    sae = TopKSAE(d_in=config.d_in, d_sae=config.d_sae, k=config.k)
    sae.w_enc.data = state["W_enc"].T.contiguous().to(dtype=dtype)
    sae.w_dec.data = state["W_dec"].T.contiguous().to(dtype=dtype)
    sae.b_enc.data = state["b_enc"].to(dtype=dtype)
    sae.b_dec.data = state["b_dec"].to(dtype=dtype)
    sae.eval()
    return sae.to(config.device)


def load_sae(config: GemmaScopeSAEConfig | QwenScopeSAEConfig) -> SAEBase:
    """Download (if needed) and load a pretrained SAE from config.

    Dispatches on the config type. Returns an ``SAEBase`` subclass —
    ``JumpReLUSAE`` for Gemma-scope, ``TopKSAE`` for Qwen-scope.
    """
    if config.dtype not in DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype '{config.dtype}'. Valid: {list(DTYPE_MAP.keys())}"
        )
    if isinstance(config, QwenScopeSAEConfig):
        return _load_qwen_scope_sae(config)
    if isinstance(config, GemmaScopeSAEConfig):
        return _load_gemma_scope_sae(config)
    raise TypeError(
        f"Unsupported SAE config type: {type(config).__name__}. "
        "Expected GemmaScopeSAEConfig or QwenScopeSAEConfig."
    )
