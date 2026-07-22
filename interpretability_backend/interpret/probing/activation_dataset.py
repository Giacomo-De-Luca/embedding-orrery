"""Storage container for collected activations + targets.

Backbone-agnostic: extractors (encoder, gemma, sae) all produce instances
of this class. Probes consume it.

This module is storage-only — the Gemma-specific collection logic that
previously lived inside `ActivationDataset.collect()` has moved to
`probing.extraction.extract_gemma_activations`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from interpret.probing.utils.matrix_ops import FeatureMatrix, select_rows


class ActivationDataset:
    """Collected activation–target pairs from model inference.

    Attributes:
        activations: Mapping from (layer, intermediate) to Tensor[N, hidden_size]
            — or a scipy CSR matrix of the same shape for sparse SAE-pooled
            extractions (``sparse: true``). `save`/`load` handle both
            (torch.save pickles scipy matrices as-is).
        targets: Tensor[N, target_dim] for regression, or LongTensor[N] for classification.
            Empty Tensor when activations are pre-target (e.g. fresh extraction
            before joining with manifest values).
        sample_ids: List of string identifiers, ordered to match dim 0 of every
            activation tensor.
        metadata: Dict of collection parameters and dataset info. May include
            `kept_by_layer` for SAE-encoded datasets — a mapping from filtered
            column index to original SAE feature index, required for downstream
            label lookup.
    """

    def __init__(
        self,
        activations: dict[tuple[int, str], FeatureMatrix] | None = None,
        targets: torch.Tensor | None = None,
        sample_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.activations = activations if activations is not None else {}
        self.targets = targets if targets is not None else torch.empty(0)
        self.sample_ids = list(sample_ids) if sample_ids is not None else []
        self.metadata = dict(metadata) if metadata is not None else {}

    def save(self, path: Path) -> Path:
        """Save dataset to disk as a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "activations": self.activations,
                "targets": self.targets,
                "sample_ids": self.sample_ids,
                "metadata": self.metadata,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> ActivationDataset:
        """Load a previously saved dataset from disk."""
        data = torch.load(Path(path), weights_only=False)
        return cls(
            activations=data["activations"],
            targets=data["targets"],
            sample_ids=data["sample_ids"],
            metadata=data["metadata"],
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def get(self, layer: int, intermediate: str) -> tuple[FeatureMatrix, torch.Tensor]:
        """Get (activations, targets) for a specific layer/intermediate."""
        key = (layer, intermediate)
        if key not in self.activations:
            available = list(self.activations.keys())
            raise KeyError(
                f"No activations for layer {layer}/{intermediate}. Available: {available}"
            )
        return self.activations[key], self.targets

    def layer_intermediate_keys(self) -> list[tuple[int, str]]:
        """List all available (layer, intermediate) combinations."""
        return list(self.activations.keys())

    def subset(self, sample_ids: list[str]) -> ActivationDataset:
        """Return a new dataset with rows filtered + reordered to match `sample_ids`.

        The returned dataset has its activation tensors and `sample_ids`
        reordered so that row `i` corresponds to `sample_ids[i]`. Targets
        are forwarded only when non-empty (preserves the pre-target
        convention used by extraction).

        Raises:
            KeyError: If any requested ID is not present in `self.sample_ids`.
        """
        id_to_idx = {sid: i for i, sid in enumerate(self.sample_ids)}
        missing = [s for s in sample_ids if s not in id_to_idx]
        if missing:
            raise KeyError(
                f"{len(missing)} sample_ids not in dataset. First 5: {missing[:5]}",
            )
        index_list = [id_to_idx[s] for s in sample_ids]
        indices = torch.tensor(index_list, dtype=torch.long)
        new_acts = {key: select_rows(mat, index_list) for key, mat in self.activations.items()}
        new_targets = self.targets[indices] if self.targets.numel() > 0 else self.targets
        new_metadata = dict(self.metadata)
        new_metadata["subset_n"] = len(sample_ids)
        return ActivationDataset(
            activations=new_acts,
            targets=new_targets,
            sample_ids=list(sample_ids),
            metadata=new_metadata,
        )
