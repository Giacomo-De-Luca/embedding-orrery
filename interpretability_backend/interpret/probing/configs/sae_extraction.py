"""SAE extraction config — first-class extraction type.

Consumes another extraction's per-sample `[N, d_in]` residuals and
encodes them through Gemma-Scope JumpReLU SAEs to per-feature
activations `[N, d_kept]`. The `token_position` field is informational
and validated against the source extraction's pooling at orchestrator
time — they MUST match because SAE encoding can't undo or change pooling.

Future extension: token-level extraction + downstream pooling would let
the SAE choose its own pooling strategy. Today's pipeline is single-token-
per-sample; SAE inherits whatever pooling the source did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from interpret.probing.utils.enums import TokenPosition


@dataclass
class SAEExtractionConfig:
    """Encode another extraction's residuals through Gemma-Scope SAEs."""

    name: str                           # required, drives cache filename + folder
    source_extraction: str              # name of the residual extraction to consume
    source_intermediate: str = "post_mlp"  # which intermediate to read from the source
    # Documentation + validation: the upstream pooling that produced the
    # source vectors. The orchestrator raises if this disagrees with the
    # source extraction's actual `token_position`.
    token_position: TokenPosition = TokenPosition.WORD_LAST

    type: Literal["sae"] = "sae"
    layers: list[int] = field(default_factory=lambda: [9, 17, 22, 29])
    width: str = "16k"
    device: str = "cpu"
    drop_dead_features: bool = True

    def __post_init__(self) -> None:
        if self.type != "sae":
            raise ValueError(
                f"SAEExtractionConfig.type must be 'sae', got {self.type!r}",
            )
        if not self.name:
            raise ValueError("SAEExtractionConfig.name is required.")
        if not self.source_extraction:
            raise ValueError(
                "SAEExtractionConfig.source_extraction is required — "
                "must reference another extraction's name.",
            )
        if isinstance(self.token_position, str):
            self.token_position = TokenPosition(self.token_position)

    def cache_filename(self) -> str:
        return self.name
