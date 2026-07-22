"""Shared stub SAE + token-level source for the sae_pooled test suites.

Imported by `test_sae_pooled_extraction.py` (dense path) and
`test_sparse_sae_probing.py` (sparse path) so both pin the same synthetic
data. Not a test module — pytest ignores it (no `test_` prefix).
"""

import torch

from interpret.probing.activation_dataset import ActivationDataset

D_IN, D_SAE = 4, 3


class StubSAE:
    """encode(x) = x[:, :3] — feature values mirror the first residual dims."""

    w_dec = torch.zeros(D_SAE, D_IN)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :D_SAE].clone()


def token_source(prepends_bos: bool = True, family: str = "gemma") -> ActivationDataset:
    """3 samples, lengths [3, 2, 1] (first position = BOS when applicable).

    BOS rows carry huge values (the activation sink); sample 2 is BOS-only
    and exercises the degenerate-range fallback.
    """
    residual = torch.tensor(
        [
            # sample 0: BOS + 2 tokens
            [100.0, 100.0, 100.0, 0.0],
            [1.0, -5.0, 0.0, 0.0],
            [3.0, 2.0, 0.0, 0.0],
            # sample 1: BOS + 1 token
            [100.0, 100.0, 100.0, 0.0],
            [7.0, -1.0, 0.0, 0.0],
            # sample 2: BOS only
            [2.0, 0.0, 5.0, 0.0],
        ],
    )
    return ActivationDataset(
        activations={(0, "resid_post"): residual, (2, "resid_post"): residual * 2},
        targets=torch.empty(0),
        sample_ids=["s0", "s1", "s2"],
        metadata={
            "extraction_type": "token_residuals",
            "token_level": True,
            "family": family,
            "token_offsets": [0, 3, 5, 6],
            "prepends_bos": prepends_bos,
            "storage_dtype": "float32",
        },
    )
