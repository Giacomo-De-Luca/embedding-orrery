"""Configuration for the refusal-direction replication pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VALID_INTERMEDIATES = {"pre_attn", "post_attn", "mlp_out", "post_mlp"}
VALID_BYPASS_MODES = {"ablation", "actadd"}


@dataclass
class RefusalConfig:
    """Settings for `RefusalRunner`.

    Defaults match the reference paper's Gemma config (n_train=128, n_val=32,
    KL threshold 0.1, 20 % top-of-stack layer pruning) and the project's
    Gemma-3-4b architecture (34 layers, d_model 2560).
    """

    model_name: str = "google/gemma-3-4b-it"
    n_layers: int = 34
    d_model: int = 2560

    intermediates: tuple[str, ...] = ("pre_attn",)
    n_train: int = 128
    n_val: int = 32
    n_test: int = 100
    n_eval: int | None = None  # subsample harmful eval dataset; None = use all

    refusal_token_ids: tuple[int, ...] = (236777,)
    # Gemma-3 SentencePiece: 236777 == "I". (Gemma 1/2 used 235285, which in
    # Gemma-3's renumbered vocab decodes to "Dated" — verified via the model
    # tokeniser. `tokens.verify_refusal_tokens` warns if this drifts again.)

    bypass_mode: str = "actadd"
    # "actadd"   → single ADDITIVE op at the source layer with coeff
    #              `actadd_bypass_coeff` (default -1). Selection criterion that
    #              works on Gemma-3-4b's post-norm residual stream.
    # "ablation" → original paper criterion: three-site projection ablation at
    #              every layer × {RESID_POST, ATTN_OUT, MLP_OUT}. Works for
    #              Gemma 1/2, Llama, Qwen-1.8B; collapses Gemma-3-4b (residual
    #              norms in the thousands → projection magnitude exceeds the
    #              ~2000 perturbation threshold; see _diagnose.py).
    actadd_bypass_coeff: float = -1.0

    kl_threshold: float = 0.1
    induce_refusal_threshold: float = 0.0
    prune_layer_pct: float = 0.20

    eval_dataset: str = "jailbreakbench"
    max_new_tokens: int = 256
    seed: int = 42

    output_dir: Path = field(
        default_factory=lambda: Path("resources/experiments/refusal_directions")
    )
    splits_dir: Path = field(
        default_factory=lambda: Path("resources/refusal_direction/splits")
    )
    eval_dir: Path = field(
        default_factory=lambda: Path("resources/refusal_direction/processed")
    )

    def __post_init__(self) -> None:
        unknown = set(self.intermediates) - VALID_INTERMEDIATES
        if unknown:
            raise ValueError(
                f"Unknown intermediates: {sorted(unknown)}. "
                f"Valid: {sorted(VALID_INTERMEDIATES)}"
            )
        if not self.intermediates:
            raise ValueError("intermediates must contain at least one entry")
        if not 0.0 <= self.prune_layer_pct < 1.0:
            raise ValueError("prune_layer_pct must be in [0, 1)")
        if self.bypass_mode not in VALID_BYPASS_MODES:
            raise ValueError(
                f"Unknown bypass_mode: {self.bypass_mode!r}. "
                f"Valid: {sorted(VALID_BYPASS_MODES)}"
            )

        self.output_dir = Path(self.output_dir)
        self.splits_dir = Path(self.splits_dir)
        self.eval_dir = Path(self.eval_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def generate_dir(self) -> Path:
        return self.output_dir / "generate_directions"

    @property
    def select_dir(self) -> Path:
        return self.output_dir / "select_direction"

    @property
    def completions_dir(self) -> Path:
        return self.output_dir / "completions"

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "intermediates": list(self.intermediates),
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_eval": self.n_eval,
            "refusal_token_ids": list(self.refusal_token_ids),
            "kl_threshold": self.kl_threshold,
            "induce_refusal_threshold": self.induce_refusal_threshold,
            "prune_layer_pct": self.prune_layer_pct,
            "eval_dataset": self.eval_dataset,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "splits_dir": str(self.splits_dir),
            "eval_dir": str(self.eval_dir),
        }
