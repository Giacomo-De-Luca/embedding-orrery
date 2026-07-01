"""Per-layer hidden-state extraction from HuggingFace encoder models.

Loads the model with `output_hidden_states=True`, processes samples in
batches, applies pooling, and returns a populated `ActivationDataset`
with keys `(layer, "hidden")`.

The previous `ActivationExtractor` class in
`encoder_probing/extractor.py` is replaced by a free function that
returns the canonical `ActivationDataset` directly.
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from interpret.probing.activation_dataset import ActivationDataset
from interpret.probing.configs.extraction import (
    EncoderExtractionConfig,
)

INTERMEDIATE_KEY = "hidden"


def extract_encoder_activations(
    config: EncoderExtractionConfig,
    samples: list[str],
) -> ActivationDataset:
    """Extract per-layer pooled activations from an HF encoder.

    Args:
        config: Model + pooling + layer selection.
        samples: Texts to encode. Order is preserved as the dataset's
            `sample_ids`, matching dim 0 of every activation tensor.

    Returns:
        Populated `ActivationDataset` with empty `targets` and metadata.
    """
    device = config.device or _autodetect_device()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = (
        AutoModel.from_pretrained(config.model_name, output_hidden_states=True)
        .to(device)
        .eval()
    )

    num_hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layers = _resolve_layers(config.layers, num_hidden_layers)

    print(
        f"extract_encoder_activations: {config.model_name} | "
        f"{num_hidden_layers} layers | hidden={hidden_size} | "
        f"extracting {layers} | pooling={config.pooling} | device={device}",
    )

    accum: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    with torch.no_grad():
        for start in tqdm(
            range(0, len(samples), config.batch_size),
            desc="Encoder activations",
        ):
            batch = samples[start : start + config.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_special_tokens_mask=True,
            ).to(device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            for layer in layers:
                hidden = outputs.hidden_states[layer]
                pooled = _pool(
                    hidden,
                    inputs["attention_mask"],
                    inputs.get("special_tokens_mask"),
                    config.pooling,
                )
                accum[layer].append(pooled.cpu().float())

    activations = {
        (layer, INTERMEDIATE_KEY): torch.cat(tensors)
        for layer, tensors in accum.items()
    }
    metadata = {
        "model_name": config.model_name,
        "pooling": config.pooling,
        "num_layers": len(activations),
        "hidden_size": hidden_size,
        "num_samples": len(samples),
        "layers": layers,
        "intermediates": [INTERMEDIATE_KEY],
        "extraction_type": "encoder",
    }
    return ActivationDataset(
        activations=activations,
        targets=torch.empty(0),
        sample_ids=list(samples),
        metadata=metadata,
    )


def _resolve_layers(
    layers: list[int] | None, num_hidden_layers: int,
) -> list[int]:
    """Auto-select first/middle/last when `layers` is None."""
    if layers is not None:
        return list(layers)
    total = num_hidden_layers + 1  # +1 for the embedding layer (index 0)
    mid = total // 2
    return sorted({0, mid, num_hidden_layers})


def _pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor | None,
    strategy: str,
) -> torch.Tensor:
    """Reduce [batch, seq_len, hidden] to [batch, hidden]."""
    if strategy == "cls":
        return hidden_states[:, 0, :]
    if strategy == "mean":
        if special_tokens_mask is not None:
            mask = attention_mask * (1 - special_tokens_mask)
        else:
            mask = attention_mask
        mask = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts
    if strategy == "last":
        seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
        batch_idx = torch.arange(
            hidden_states.size(0), device=hidden_states.device,
        )
        return hidden_states[batch_idx, seq_lengths, :]
    raise ValueError(f"Unknown pooling strategy: {strategy!r}")


def _autodetect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
