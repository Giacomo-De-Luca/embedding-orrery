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
    WIDTH_TO_D_SAE,
    GemmaScopeSAEConfig,
    QwenScopeSAEConfig,
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
        path,
        d_in=config.d_in,
        d_sae=WIDTH_TO_D_SAE[config.width],
        device=config.device,
        dtype=dtype,
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

    # On-disk W_enc is (d_sae, d_in); the tensor is the source of truth for the
    # dims, so a new Qwen model needs no d_in/d_sae bookkeeping. Validate the
    # advisory config d_in against it to catch a wrong model_size early. d_sae
    # is left to the tensor (config.d_sae is only the smoke-test's estimate).
    d_sae_disk, d_in_disk = state["W_enc"].shape
    if config.d_in is not None and config.d_in != d_in_disk:
        raise ValueError(
            f"d_in mismatch for {config.repo_id}: config expects {config.d_in}, "
            f"weights have {d_in_disk}. Check model_size."
        )

    sae = TopKSAE(d_in=d_in_disk, d_sae=d_sae_disk, k=config.k)
    sae.w_enc.data = state["W_enc"].T.contiguous().to(dtype=dtype)
    sae.w_dec.data = state["W_dec"].T.contiguous().to(dtype=dtype)
    sae.b_enc.data = state["b_enc"].to(dtype=dtype)
    sae.b_dec.data = state["b_dec"].to(dtype=dtype)
    sae.eval()
    return sae.to(config.device)


# Module-level cache of loaded SAEs, keyed by the config fields that determine
# on-disk identity and post-load placement. Survives across HookManager
# instances within a process, so repeat steering / probing calls reuse the
# already-resident MPS tensors instead of re-reading ~320 MB of safetensors
# off disk every time. Cleared by ``clear_sae_cache()`` — call from
# ``InterpretService.unload_model()`` so a model variant switch doesn't
# retain stale device tensors.
_SAE_CACHE: dict[tuple, SAEBase] = {}


def _config_to_key(config: GemmaScopeSAEConfig | QwenScopeSAEConfig) -> tuple:
    """Build a hashable cache key from the fields that identify the weights.

    Includes ``dtype`` and ``device`` so a CPU request never returns an MPS
    tensor (or vice versa). Excludes hook-policy fields (``read_only``,
    ``prefill_only``, ``collect_last_only``) — the same loaded SAE is safely
    reusable across different hook policies.
    """
    if isinstance(config, GemmaScopeSAEConfig):
        return (
            GemmaScopeSAEConfig,
            config.layer_index,
            config.hook_type.value,
            config.width,
            config.model_size,
            config.variant,
            config.l0_size,
            config.dtype,
            config.device,
        )
    if isinstance(config, QwenScopeSAEConfig):
        return (
            QwenScopeSAEConfig,
            config.layer_index,
            config.hook_type.value,
            config.model_size,
            config.width,
            str(config.k),
            config.dtype,
            config.device,
        )
    raise TypeError(
        f"Unsupported SAE config type: {type(config).__name__}. "
        "Expected GemmaScopeSAEConfig or QwenScopeSAEConfig."
    )


def clear_sae_cache() -> None:
    """Drop all cached SAEs so MPS / CUDA memory can be reclaimed."""
    _SAE_CACHE.clear()


def load_sae(config: GemmaScopeSAEConfig | QwenScopeSAEConfig) -> SAEBase:
    """Download (if needed) and load a pretrained SAE from config.

    Dispatches on the config type. Returns an ``SAEBase`` subclass —
    ``JumpReLUSAE`` for Gemma-scope, ``TopKSAE`` for Qwen-scope.

    Loaded SAEs are cached at module level by ``_config_to_key(config)``;
    repeat calls with an equivalent config return the same in-memory
    instance. Call ``clear_sae_cache()`` to free.
    """
    if config.dtype not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{config.dtype}'. Valid: {list(DTYPE_MAP.keys())}")
    key = _config_to_key(config)
    cached = _SAE_CACHE.get(key)
    if cached is not None:
        return cached
    if isinstance(config, QwenScopeSAEConfig):
        sae = _load_qwen_scope_sae(config)
    elif isinstance(config, GemmaScopeSAEConfig):
        sae = _load_gemma_scope_sae(config)
    else:
        raise TypeError(
            f"Unsupported SAE config type: {type(config).__name__}. "
            "Expected GemmaScopeSAEConfig or QwenScopeSAEConfig."
        )
    _SAE_CACHE[key] = sae
    return sae
