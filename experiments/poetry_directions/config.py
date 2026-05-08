"""Configuration for the poetry-direction experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VALID_INTERMEDIATES = {"pre_attn", "post_attn", "mlp_out", "post_mlp"}


EXPERIMENTS: dict[str, dict] = {
    "poems_paraphrase": {
        "loader": "pairs",
        "args": {},
    },
    "poetry_prose": {
        "loader": "filtered",
        "args": {
            "class_a_filter": {"type": "poetry"},
            "class_b_filter": {"type": "prose"},
            "text_column": "prompt",
        },
    },
    "poetry_unsafe_safe": {
        "loader": "filtered",
        "args": {
            "class_a_filter": {"type": "poetry", "final_safety_status": "unsafe"},
            "class_b_filter": {"type": "poetry", "final_safety_status": "safe"},
            "text_column": "prompt",
        },
    },
}


@dataclass
class PoetryConfig:
    """Settings for `PoetryRunner`.

    Defaults mirror `RefusalConfig` so the two experiments' results are
    directly comparable. The bypass coefficient grid covers both signs because
    the poetry direction's bypass sign is unknown a priori (unlike the refusal
    direction, where -1 is "subtract refusal").
    """

    name: str

    model_name: str = "google/gemma-3-4b-it"
    n_layers: int = 34
    d_model: int = 2560
    intermediates: tuple[str, ...] = ("pre_attn", "post_attn", "mlp_out")
    # All three captured in a single forward pass per prompt (extract.py
    # batches via cache_activations). Choices:
    #   pre_attn  — residual stream at layer entry
    #   post_attn — residual after attn write, before mlp (within-layer)
    #   mlp_out   — MLP block's raw output (NOT residual stream — a different signal)
    # `post_mlp` is intentionally omitted: post_mlp(L) ≡ pre_attn(L+1), and
    # post_mlp(33) falls in the pruned top-20% layer band → never a candidate.
    max_per_class: int | None = None

    n_val: int = 32
    prune_layer_pct: float = 0.20
    kl_threshold: float = 0.5
    coefficients: tuple[float, ...] = (-1.0, +1.0)
    positions: tuple[int, ...] | None = None
    # Which EOI positions (negative indices) to sweep. None = all positions
    # captured during extraction (typically 5: -5..-1). Restrict to e.g.
    # `(-1, -3)` to cut sweep cost proportionally; full sweep is
    # recommended when compute permits. The sweep supports resume: if you
    # change `positions` between runs, already-computed cells are reused
    # from `direction_evaluations.csv` and only new ones are computed.
    magnitude_cap: float = 1800.0
    # Skip sweep cells where |coeff| * |v| > magnitude_cap. Gemma-3 collapses
    # at perturbations above ~2000 (see refusal_steer_tester.ipynb comment); 1800
    # leaves a small safety margin. Skipped cells are written as NaN, so the
    # filter discards them. Set to math.inf to disable.
    refusal_token_ids: tuple[int, ...] = (236777,)

    eval_dataset: str = "jailbreakbench"
    n_eval: int | None = None
    max_new_tokens: int = 256

    seed: int = 42

    output_dir_root: Path = field(
        default_factory=lambda: Path("resources/experiments/poetry_directions")
    )
    poems_csv: Path = field(
        default_factory=lambda: Path(
            "resources/poetry/paraphrased_poems_aligned.csv"
        )
    )
    prompts_csv: Path = field(
        default_factory=lambda: Path(
            "resources/poetry/alignment_dataset_long_clean.tsv"
        )
    )
    splits_dir: Path = field(
        default_factory=lambda: Path("resources/refusal_direction/splits")
    )
    eval_dir: Path = field(
        default_factory=lambda: Path("resources/refusal_direction/processed")
    )

    def __post_init__(self) -> None:
        if self.name not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment name: {self.name!r}. "
                f"Valid: {sorted(EXPERIMENTS)}"
            )
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
        if not self.coefficients:
            raise ValueError("coefficients must contain at least one value")

        self.output_dir_root = Path(self.output_dir_root)
        self.poems_csv = Path(self.poems_csv)
        self.prompts_csv = Path(self.prompts_csv)
        self.splits_dir = Path(self.splits_dir)
        self.eval_dir = Path(self.eval_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self.output_dir_root / self.name

    @property
    def extract_dir(self) -> Path:
        return self.output_dir / "extract"

    @property
    def sweep_dir(self) -> Path:
        return self.output_dir / "sweep"

    @property
    def completions_dir(self) -> Path:
        return self.output_dir / "completions"

    @property
    def experiment_spec(self) -> dict:
        return EXPERIMENTS[self.name]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "intermediates": list(self.intermediates),
            "max_per_class": self.max_per_class,
            "n_val": self.n_val,
            "prune_layer_pct": self.prune_layer_pct,
            "kl_threshold": self.kl_threshold,
            "coefficients": list(self.coefficients),
            "positions": list(self.positions) if self.positions is not None else None,
            "magnitude_cap": self.magnitude_cap,
            "refusal_token_ids": list(self.refusal_token_ids),
            "eval_dataset": self.eval_dataset,
            "n_eval": self.n_eval,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "poems_csv": str(self.poems_csv),
            "prompts_csv": str(self.prompts_csv),
            "splits_dir": str(self.splits_dir),
            "eval_dir": str(self.eval_dir),
            "experiment_spec": self.experiment_spec,
        }
