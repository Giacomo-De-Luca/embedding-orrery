"""Per-layer activation extraction from Gemma3 via the PyTorch wrapper.

Two entry points:
  * `extract_gemma_activations(config, samples, model)` — text mode.
    Iterates the manifest's full `samples` list. The orchestrator passes
    `manifest.samples` directly so activations align with
    `manifest.get_sample_indices(...)` for downstream filtering.
  * `extract_gemma_activations_from_dataframe(config, manifest_df, model)`
    — multimodal / image mode. Iterates DataFrame rows so per-row
    metadata (image filename, etc.) is available. Use `image_column`
    set on the config.

Both produce one `[hidden]` vector per sample, pooled per
`config.token_position`. Downstream consumers can't re-pool. See
`TokenPosition` docstring for the future token-level extension.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.extraction import (
    GemmaExtractionConfig,
)
from interpret.probing.utils.enums import TokenPosition


def extract_gemma_activations(
    config: GemmaExtractionConfig,
    samples: list[str],
    model: GemmaPytorchInference,  # noqa: F821 — avoid heavy import
) -> ActivationDataset:
    """Run inference over `samples` (text mode) and collect activations.

    Args:
        config: Layers, intermediates, token position, prompt template.
            Must have `prompt_column` set (text mode).
        samples: Ordered list of input strings — one prompt per sample.
            Order is preserved as `dataset.sample_ids`, matching dim 0
            of every activation tensor.
        model: Loaded `GemmaPytorchInference` instance.

    Returns:
        Populated `ActivationDataset` with empty `targets`.
    """
    if config.image_column is not None:
        raise ValueError(
            "extract_gemma_activations: image_column is set; use "
            "extract_gemma_activations_from_dataframe for image mode.",
        )
    if not samples:
        raise ValueError("samples is empty.")

    use_word_last = config.token_position is TokenPosition.WORD_LAST

    layers_set = set(config.layers)
    intermediates_set = set(config.intermediates)
    act_accum: dict[tuple[int, str], list[torch.Tensor]] = {
        (layer, inter): []
        for layer in config.layers
        for inter in config.intermediates
    }

    try:
        model.configure_cache(
            layers=layers_set,
            intermediates=intermediates_set,
            prefill=(config.cache_phase == "prefill"),
            last=(config.cache_phase == "last"),
        )
        for sample in tqdm(samples, desc="Gemma activations"):
            model.reset_prefill_cache()
            token_index: int | None = None

            if use_word_last:
                formatted = config.prompt_template.format(word=sample)
                # The +1 below is for the BOS token prepended by the tokenizer.
                prefix = config.prompt_template.split("{word}")[0]
                prefix_len = len(model.tokenize(prefix, bos=False)) + 1
                word_len = len(model.tokenize(sample, bos=False))
                token_index = prefix_len + word_len - 1
                model.generate_from_template(formatted, output_len=1)
            else:
                model.generate(sample, output_len=1)

            cache = model.get_cached_activations()
            phase_cache = cache.get(config.cache_phase, {})

            for layer in config.layers:
                layer_cache = phase_cache.get(layer, {})
                for inter in config.intermediates:
                    act = layer_cache.get(inter)
                    if act is None:
                        raise RuntimeError(
                            f"No activation for layer {layer}/{inter}. "
                            f"Cache keys: {list(phase_cache.keys())}",
                        )
                    reduced = _reduce_sequence(
                        act, config.token_position, token_index=token_index,
                    )
                    act_accum[(layer, inter)].append(reduced.float())
    finally:
        model.clear_cache()

    activations = {
        key: torch.stack(tensors) for key, tensors in act_accum.items()
    }
    metadata = _build_metadata(config, n=len(samples))
    return ActivationDataset(
        activations=activations,
        targets=torch.empty(0),
        sample_ids=list(samples),
        metadata=metadata,
    )


def extract_gemma_activations_from_dataframe(
    config: GemmaExtractionConfig,
    manifest_df: pd.DataFrame,
    model: GemmaPytorchInference,  # noqa: F821
) -> ActivationDataset:
    """Image / multimodal mode: iterate DataFrame rows, use `image_column`."""
    if config.image_column is None:
        raise ValueError(
            "image_column is not set — use extract_gemma_activations "
            "for text mode.",
        )
    if len(manifest_df) == 0:
        raise ValueError("manifest_df is empty.")

    layers_set = set(config.layers)
    intermediates_set = set(config.intermediates)
    act_accum: dict[tuple[int, str], list[torch.Tensor]] = {
        (layer, inter): []
        for layer in config.layers
        for inter in config.intermediates
    }
    sample_ids: list[str] = []

    try:
        model.configure_cache(
            layers=layers_set,
            intermediates=intermediates_set,
            prefill=(config.cache_phase == "prefill"),
            last=(config.cache_phase == "last"),
        )
        for _, row in tqdm(
            manifest_df.iterrows(),
            total=len(manifest_df),
            desc="Gemma activations (image)",
        ):
            model.reset_prefill_cache()
            image_path = _resolve_image_path(
                row[config.image_column],
                Path(config.image_dir) if config.image_dir else None,
            )
            prompt = config.prompt_template or ""
            model.generate_with_image(prompt, str(image_path), output_len=1)
            sample_id = str(row[config.image_column])
            sample_ids.append(sample_id)

            cache = model.get_cached_activations()
            phase_cache = cache.get(config.cache_phase, {})
            for layer in config.layers:
                layer_cache = phase_cache.get(layer, {})
                for inter in config.intermediates:
                    act = layer_cache.get(inter)
                    if act is None:
                        raise RuntimeError(
                            f"No activation for layer {layer}/{inter}. "
                            f"Cache keys: {list(phase_cache.keys())}",
                        )
                    reduced = _reduce_sequence(
                        act, config.token_position, token_index=None,
                    )
                    act_accum[(layer, inter)].append(reduced.float())
    finally:
        model.clear_cache()

    activations = {
        key: torch.stack(tensors) for key, tensors in act_accum.items()
    }
    metadata = _build_metadata(config, n=len(sample_ids))
    return ActivationDataset(
        activations=activations,
        targets=torch.empty(0),
        sample_ids=sample_ids,
        metadata=metadata,
    )


def _build_metadata(config: GemmaExtractionConfig, n: int) -> dict:
    return {
        "num_samples": n,
        "layers": list(config.layers),
        "intermediates": list(config.intermediates),
        "token_position": config.token_position.value,
        "cache_phase": config.cache_phase,
        "hidden_size": config.hidden_size,
        "extraction_type": "gemma",
    }


def _reduce_sequence(
    activation: torch.Tensor,
    strategy: TokenPosition,
    token_index: int | None = None,
) -> torch.Tensor:
    """Reduce [1, seq_len, hidden_size] to [hidden_size]."""
    act = activation.squeeze(0)
    match strategy:
        case TokenPosition.LAST:
            return act[-1]
        case TokenPosition.FIRST:
            return act[0]
        case TokenPosition.MEAN:
            return act.mean(dim=0)
        case TokenPosition.MAX:
            return act.max(dim=0).values
        case TokenPosition.WORD_LAST:
            if token_index is None:
                raise ValueError("WORD_LAST requires token_index.")
            return act[token_index]
        case _:
            raise ValueError(f"Unknown token position: {strategy}")


def _resolve_image_path(
    filename: str, image_dir: Path | None,
) -> Path:
    """Resolve an image filename to an absolute path."""
    if image_dir is not None:
        return image_dir / filename
    return Path(filename)
