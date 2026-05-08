"""Storage for SAE feature activations captured during inference."""

from dataclasses import dataclass

import torch

from interpret.sae.sae_config import HookType


@dataclass
class ActivationRecord:
    """Activations from a single forward pass through one SAE.

    Attributes:
        step: Forward pass index (0 = prefill, 1+ = decode steps).
        feature_acts: Sparse feature activations, shape (batch, seq_len, d_sae).
        reconstruction: SAE reconstruction, shape (batch, seq_len, d_in). None if not stored.
    """

    step: int
    feature_acts: torch.Tensor
    reconstruction: torch.Tensor | None = None


class ActivationStore:
    """Stores SAE feature activations captured by hooks during inference.

    Activations are keyed by (layer_index, hook_type) and accumulate
    as a list of ActivationRecords across forward passes.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[int, HookType], list[ActivationRecord]] = {}
        self._steps: dict[tuple[int, HookType], int] = {}

    def record(
        self,
        key: tuple[int, HookType],
        feature_acts: torch.Tensor,
        reconstruction: torch.Tensor | None = None,
        collect_last_only: bool = False,
    ) -> None:
        """Record activations from a forward pass.

        Step counter is tracked per (layer, hook_type) key, so multi-SAE
        setups get contiguous step numbers per SAE.

        Args:
            key: (layer_index, hook_type) identifying the SAE.
            feature_acts: Sparse feature activations tensor.
            reconstruction: Optional SAE reconstruction tensor.
            collect_last_only: If True, only keep the most recent record.
        """
        step = self._steps.get(key, 0)
        entry = ActivationRecord(step, feature_acts, reconstruction)
        if collect_last_only:
            self._store[key] = [entry]
        else:
            self._store.setdefault(key, []).append(entry)
        self._steps[key] = step + 1

    def get(
        self, layer: int, hook_type: HookType = HookType.RESID_POST
    ) -> list[ActivationRecord]:
        """Get all activation records for a layer."""
        return self._store.get((layer, hook_type), [])

    def prefill(
        self, layer: int, hook_type: HookType = HookType.RESID_POST
    ) -> ActivationRecord | None:
        """Get the prefill activation record (step 0) for a layer.

        The prefill is the first forward pass, which processes the full
        prompt in one go (seq_len > 1). Decode steps have seq_len == 1.
        """
        records = self.get(layer, hook_type)
        return records[0] if records else None

    def latest(
        self, layer: int, hook_type: HookType = HookType.RESID_POST
    ) -> ActivationRecord | None:
        """Get the most recent activation record for a layer."""
        records = self.get(layer, hook_type)
        return records[-1] if records else None

    def all_feature_acts(
        self, layer: int, hook_type: HookType = HookType.RESID_POST
    ) -> torch.Tensor:
        """Concatenate all feature activations along the sequence dimension.

        Returns:
            Tensor of shape (batch, total_seq_len, d_sae).

        Raises:
            ValueError: If no activations are stored for the given layer.
        """
        records = self.get(layer, hook_type)
        if not records:
            raise ValueError(
                f"No activations stored for layer {layer}, hook_type {hook_type}"
            )
        return torch.cat([r.feature_acts for r in records], dim=1)

    def clear(self) -> None:
        """Remove all stored activations and reset step counters."""
        self._store.clear()
        self._steps.clear()
