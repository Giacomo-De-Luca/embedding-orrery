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


StoreKey = tuple[int, HookType, str]


class ActivationStore:
    """Stores SAE feature activations captured by hooks during inference.

    Activations are keyed by ``(layer_index, hook_type, sae_id)`` and
    accumulate as a list of ActivationRecords across forward passes.
    ``sae_id`` defaults to ``""`` so single-SAE callers — both writers
    and readers — work unchanged. The extra slot disambiguates two
    SAEs co-attached at the same ``(layer, hook_type)`` site (e.g.
    Gemma L29 W16K and L29 W65K in one forward pass).
    """

    def __init__(self) -> None:
        self._store: dict[StoreKey, list[ActivationRecord]] = {}
        self._steps: dict[StoreKey, int] = {}

    def record(
        self,
        key: StoreKey,
        feature_acts: torch.Tensor,
        reconstruction: torch.Tensor | None = None,
        collect_last_only: bool = False,
    ) -> None:
        """Record activations from a forward pass.

        Step counter is tracked per ``(layer, hook_type, sae_id)`` key, so
        multi-SAE setups (including same-site co-attachment) get
        contiguous, non-overlapping step numbers per SAE.

        Args:
            key: ``(layer_index, hook_type, sae_id)`` identifying the SAE.
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
        self,
        layer: int,
        hook_type: HookType = HookType.RESID_POST,
        sae_id: str = "",
    ) -> list[ActivationRecord]:
        """Get all activation records for a layer (and optional sae_id)."""
        return self._store.get((layer, hook_type, sae_id), [])

    def prefill(
        self,
        layer: int,
        hook_type: HookType = HookType.RESID_POST,
        sae_id: str = "",
    ) -> ActivationRecord | None:
        """Get the prefill activation record (step 0) for a layer.

        The prefill is the first forward pass, which processes the full
        prompt in one go (seq_len > 1). Decode steps have seq_len == 1.
        """
        records = self.get(layer, hook_type, sae_id)
        return records[0] if records else None

    def latest(
        self,
        layer: int,
        hook_type: HookType = HookType.RESID_POST,
        sae_id: str = "",
    ) -> ActivationRecord | None:
        """Get the most recent activation record for a layer."""
        records = self.get(layer, hook_type, sae_id)
        return records[-1] if records else None

    def all_feature_acts(
        self,
        layer: int,
        hook_type: HookType = HookType.RESID_POST,
        sae_id: str = "",
    ) -> torch.Tensor:
        """Concatenate all feature activations along the sequence dimension.

        Returns:
            Tensor of shape (batch, total_seq_len, d_sae).

        Raises:
            ValueError: If no activations are stored for the given layer.
        """
        records = self.get(layer, hook_type, sae_id)
        if not records:
            raise ValueError(
                f"No activations stored for layer {layer}, hook_type "
                f"{hook_type}, sae_id={sae_id!r}"
            )
        return torch.cat([r.feature_acts for r in records], dim=1)

    def clear(self) -> None:
        """Remove all stored activations and reset step counters."""
        self._store.clear()
        self._steps.clear()
