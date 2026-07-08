"""Per-layer residual-stream norm profiler for Gemma / Qwen wrappers.

Measures ``||h_L||`` — the L2 norm of the residual stream at each decoder
layer's ``RESID_POST`` site — by running test prompts through a wrapper in
prefill and reading the hidden states each wrapper already captures.

Why this exists: additive steering adds ``strength * v`` to the residual
stream ``h`` at layer ``L`` (see ``interpret/sae/steering.py::apply_steering``).
How disruptive that is depends on the dimensionless ratio

    rho = strength * ||v|| / ||h_L||

so ``||h_L||`` is the model-level, prompt-stable *denominator* that turns a
raw steering coefficient into a meaningful "fraction of the residual stream".
Gemma-scope SAE decoder rows are unit-norm (``||v|| == 1``), so for SAE
features ``rho == strength / ||h_L||`` and this per-layer table is a complete
steering-strength hint; only pre-extracted direction vectors carry their own
``||v||``.

No new hooks: this reuses each wrapper's existing ``cache_activations``
context manager. The two families return different capture-dict shapes and
devices, reconciled by ``_capture_to_layer_tensors``:

    Gemma: {"prefill": {L: {"post_mlp": T[1, seq, d]}}}  (tensors on CPU)
    Qwen:  {L: {HookType.RESID_POST: T[1, seq, d]}}       (tensors on device)

``post_mlp`` is the ``RESID_POST`` equivalent — the full decoder-layer
output, i.e. the exact site steering targets.

The pure helpers (``compute_token_norms``, ``summarize_layer_norms``,
``residual_norms_from_capture``) take plain tensors / capture dicts and are
unit-tested without loading a model.
"""

from collections.abc import Sequence

import torch

from interpret.sae.sae_config import HookType

# Gemma intermediate name whose captured tensor equals the RESID_POST site.
_GEMMA_RESID_INTERMEDIATE = "post_mlp"


def compute_token_norms(tensor: torch.Tensor, drop_bos: bool) -> torch.Tensor:
    """Per-token L2 norm of a captured residual tensor.

    Args:
        tensor: ``[1, seq, d]`` (or ``[seq, d]``) hidden state. Batch is
            always 1 — the inference wrappers never batch.
        drop_bos: Drop position 0 (the BOS / attention-sink token, which
            carries an anomalously large norm unrepresentative of the
            content positions steering broadcasts over).

    Returns:
        1-D fp32 tensor of per-token norms, on CPU.
    """
    t = tensor.to(torch.float32)
    if t.dim() == 3:
        # [1, seq, d] -> [seq, d]; B is always 1 so this drops the batch axis.
        t = t.reshape(-1, t.shape[-1])
    norms = torch.linalg.vector_norm(t, dim=-1)  # [seq]
    if drop_bos and norms.shape[0] > 1:
        norms = norms[1:]
    return norms.detach().cpu()


def _capture_to_layer_tensors(cache: dict) -> dict[int, torch.Tensor]:
    """Normalise a Gemma- or Qwen-shaped capture dict to ``{layer: tensor}``.

    Reads only the ``RESID_POST`` site. Non-layer entries (e.g. Gemma's
    top-level ``"final_norm"`` inside a step dict) are ignored.
    """
    if "prefill" in cache:  # Gemma shape
        step = cache["prefill"]
        return {
            layer: sites[_GEMMA_RESID_INTERMEDIATE]
            for layer, sites in step.items()
            if isinstance(layer, int) and _GEMMA_RESID_INTERMEDIATE in sites
        }
    # Qwen shape: {layer: {HookType: tensor}}
    out: dict[int, torch.Tensor] = {}
    for layer, sites in cache.items():
        if not isinstance(layer, int):
            continue
        for hook, tensor in sites.items():
            key = hook.value if isinstance(hook, HookType) else str(hook)
            if key == HookType.RESID_POST.value:
                out[layer] = tensor
    return out


def residual_norms_from_capture(cache: dict, drop_bos: bool) -> dict[int, torch.Tensor]:
    """Per-token residual norms per layer from one capture dict (either family)."""
    layer_tensors = _capture_to_layer_tensors(cache)
    return {layer: compute_token_norms(tensor, drop_bos) for layer, tensor in layer_tensors.items()}


def summarize_layer_norms(
    per_layer_norms: dict[int, torch.Tensor],
) -> dict[int, dict[str, float]]:
    """Reduce pooled per-token norms to summary stats per layer.

    Args:
        per_layer_norms: ``{layer: 1-D tensor}`` of per-token norms pooled
            across all prompts.

    Returns:
        ``{layer: {"median","p25","p75","mean","count"}}`` — plain floats,
        JSON-serialisable, sorted by layer. Empty layers are dropped.
    """
    quantile_points = torch.tensor([0.25, 0.5, 0.75])
    out: dict[int, dict[str, float]] = {}
    for layer in sorted(per_layer_norms):
        values = per_layer_norms[layer].to(torch.float32).flatten()
        if values.numel() == 0:
            continue
        q = torch.quantile(values, quantile_points)
        out[layer] = {
            "p25": float(q[0]),
            "median": float(q[1]),
            "p75": float(q[2]),
            "mean": float(values.mean()),
            "count": int(values.numel()),
        }
    return out


class ResidualNormProfiler:
    """Profiles per-layer residual-stream norms for a Gemma or Qwen wrapper.

    Family-agnostic: relies only on the shared ``decoder_layers`` /
    ``prepends_bos`` / ``generate`` / ``cache_activations`` contract both
    wrappers implement. Gemma is distinguished from Qwen by its
    ``configure_cache`` method (the Gemma cache is fork-internal, not
    PyTorch forward hooks).
    """

    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper
        self.n_layers = len(wrapper.decoder_layers)
        self.drop_bos = bool(wrapper.prepends_bos)
        # Gemma exposes configure_cache(); the Qwen wrapper does not.
        self._is_gemma = hasattr(wrapper, "configure_cache")
        self.d_model: int | None = None

    def _cache_context(self):
        """Open the wrapper's capture context for all layers at RESID_POST."""
        layers = set(range(self.n_layers))
        if self._is_gemma:
            return self.wrapper.cache_activations(
                layers=layers, intermediates={_GEMMA_RESID_INTERMEDIATE}
            )
        return self.wrapper.cache_activations(layers=layers, hook_types={HookType.RESID_POST})

    def profile(self, prompts: Sequence[str], output_len: int = 1) -> dict[int, dict[str, float]]:
        """Run each prompt through the model and summarise per-layer norms.

        Prompts go through the wrapper's standard chat-templated ``generate``
        (matching how steering runs); ``output_len=1`` since only the prompt
        prefill is read. Per-token norms are pooled across all prompts before
        summarising.

        Note: the pooled norms span the full templated sequence (control tokens
        like ``<start_of_turn>`` included) minus BOS (position 0, masked only
        when ``prepends_bos`` — Qwen has none, so its first-position attention
        sink is kept). The reported median is robust to this; the hint is
        advisory, so no finer token filtering is applied.
        """
        accum: dict[int, list[torch.Tensor]] = {}
        for prompt in prompts:
            with self._cache_context() as get_cache:
                self.wrapper.generate(prompt, output_len=output_len)
                cache = get_cache()
            layer_tensors = _capture_to_layer_tensors(cache)
            if self.d_model is None and layer_tensors:
                self.d_model = int(next(iter(layer_tensors.values())).shape[-1])
            for layer, tensor in layer_tensors.items():
                accum.setdefault(layer, []).append(compute_token_norms(tensor, self.drop_bos))
        pooled = {layer: torch.cat(chunks) for layer, chunks in accum.items()}
        return summarize_layer_norms(pooled)
