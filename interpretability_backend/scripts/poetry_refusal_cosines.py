"""Geometry of the poetry directions relative to the refusal direction.

Computes two cosine similarities and one orthogonal decomposition, all in
the Gemma-3-4b-it residual space (d_model = 2560):

1. ``cos(diffmeans_poetry, refusal_dir)`` — the diff-of-means "poetry/prose"
   steering vector (``resources/directions/poetry_direction.pt``) against the
   diff-of-means refusal vector (``resources/directions/refusal_direction.pt``).
   Hypothesis: substantially non-zero — the poetry steering vector carries a
   refusal-parallel component.
2. ``cos(sae_poetry, refusal_dir)`` — the Gemma-scope SAE poetry feature
   decoder vector (layer 9, 16k SAE, feature 3289) against the refusal vector.
   Hypothesis: ≈ 0 — the SAE feature is a "clean" poetry direction with no
   refusal component.

It then splits the poetry vector into the part parallel to the refusal
direction plus the orthogonal remainder::

    diffmeans_poetry = proj_onto_refusal + orthogonal_remainder

and (when ``SAVE_PIECES``) writes both pieces rescaled to a common norm so
they can be steered with separately at matched magnitude.

Note on layers: the three vectors were extracted at different sites
(poetry @ L11 post_attn, refusal @ L14 pre_attn, SAE @ L9 resid_post). Cosine
is a purely geometric comparison in the shared 2560-d residual space; it does
not assume the vectors live at the same layer.

Run with::

    uv run python -m interpretability_backend.scripts.poetry_refusal_cosines
"""

import json
import sys
from pathlib import Path

import torch

# The interpret/ toolkit uses `interpret.*` absolute imports internally, so
# `interpretability_backend/` must be on sys.path (mirrors interpret_service).
_INTERPRET_PARENT = str(Path(__file__).resolve().parents[1])
if _INTERPRET_PARENT not in sys.path:
    sys.path.insert(0, _INTERPRET_PARENT)

from interpret.sae import SAEConfig, load_sae  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIRECTIONS_DIR = Path(__file__).resolve().parents[1] / "resources" / "directions"
POETRY_PATH = DIRECTIONS_DIR / "poetry_direction.pt"
REFUSAL_PATH = DIRECTIONS_DIR / "refusal_direction.pt"

# Gemma-scope SAE holding the poetry feature.
SAE_LAYER = 9
SAE_WIDTH = "16k"
MODEL_SIZE = "4b"
VARIANT = "it"
POETRY_SAE_FEATURE = 3289

# Write the matched-norm decomposition pieces to disk so they can be steered
# with separately. Pieces are rescaled to the full poetry vector's norm.
SAVE_PIECES = True
PIECES_DIR = DIRECTIONS_DIR / "decomposition"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_direction(path: Path) -> torch.Tensor:
    """Load a 1-D direction vector as float32 on CPU."""
    vec = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(vec, torch.Tensor) or vec.ndim != 1:
        raise ValueError(
            f"{path.name} must be a 1-D tensor, got "
            f"{type(vec).__name__} shape={getattr(vec, 'shape', None)}"
        )
    return vec.to(torch.float32).contiguous()


def load_sae_feature_vector() -> torch.Tensor:
    """Load the SAE decoder vector w_dec[feature] for the poetry feature."""
    config = SAEConfig(
        layer_index=SAE_LAYER,
        width=SAE_WIDTH,
        model_size=MODEL_SIZE,
        variant=VARIANT,
        device="cpu",
    )
    sae = load_sae(config)
    # w_dec: (d_sae, d_in) in the normalised Gemma convention.
    return sae.w_dec[POETRY_SAE_FEATURE].detach().float().cpu().contiguous()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two 1-D vectors, computed in float32."""
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    poetry = load_direction(POETRY_PATH)
    refusal = load_direction(REFUSAL_PATH)
    sae_poetry = load_sae_feature_vector()

    if not (poetry.shape == refusal.shape == sae_poetry.shape):
        raise ValueError(
            "vectors must share a dimension: "
            f"poetry={tuple(poetry.shape)} refusal={tuple(refusal.shape)} "
            f"sae_poetry={tuple(sae_poetry.shape)}"
        )

    rule = "=" * 72
    print(rule)
    print("Vector norms (L2)")
    print(rule)
    print(f"  |diffmeans_poetry|       = {poetry.norm().item():.4f}")
    print(f"  |refusal_dir|            = {refusal.norm().item():.4f}")
    print(f"  |sae_poetry (w_dec[{POETRY_SAE_FEATURE}])| = {sae_poetry.norm().item():.4f}")

    print()
    print(rule)
    print("Cosine similarities")
    print(rule)
    cos_poetry_refusal = cosine(poetry, refusal)
    cos_sae_refusal = cosine(sae_poetry, refusal)
    cos_poetry_sae = cosine(poetry, sae_poetry)
    print(f"  cos(diffmeans_poetry, refusal_dir) = {cos_poetry_refusal:+.4f}   "
          f"(predicted: substantially non-zero)")
    print(f"  cos(sae_poetry,       refusal_dir) = {cos_sae_refusal:+.4f}   "
          f"(predicted: ≈ 0)")
    print(f"  cos(diffmeans_poetry, sae_poetry)  = {cos_poetry_sae:+.4f}   (for reference)")

    # --- Decomposition: poetry = (proj onto refusal) + (orthogonal remainder) ---
    refusal_hat = refusal / refusal.norm()
    proj_scalar = torch.dot(poetry, refusal_hat)          # signed length along refusal
    proj_vec = proj_scalar * refusal_hat                  # refusal-parallel component
    remainder = poetry - proj_vec                         # orthogonal "pure poetry"

    print()
    print(rule)
    print("Decomposition  diffmeans_poetry = proj_onto_refusal + orthogonal_remainder")
    print(rule)
    print(f"  signed projection length onto refusal_hat = {proj_scalar.item():+.4f}")
    print(f"  |proj_onto_refusal|     = {proj_vec.norm().item():.4f}")
    print(f"  |orthogonal_remainder|  = {remainder.norm().item():.4f}")
    frac = proj_vec.norm().item() / poetry.norm().item()
    print(f"  fraction of |poetry| along refusal = {frac:.4f}")
    # Sanity: pieces sum back, and the remainder is orthogonal to refusal.
    recon_err = (proj_vec + remainder - poetry).norm().item()
    print(f"  reconstruction error |proj + remainder - poetry| = {recon_err:.2e}")
    print(f"  cos(orthogonal_remainder, refusal_dir) = {cosine(remainder, refusal):+.2e}  "
          f"(should be ≈ 0)")

    if SAVE_PIECES:
        target_norm = poetry.norm()
        parallel_matched = proj_vec / proj_vec.norm() * target_norm
        remainder_matched = remainder / remainder.norm() * target_norm

        PIECES_DIR.mkdir(parents=True, exist_ok=True)
        parallel_path = PIECES_DIR / "poetry_refusal_parallel.pt"
        remainder_path = PIECES_DIR / "poetry_orthogonal.pt"
        torch.save(parallel_matched.contiguous(), parallel_path)
        torch.save(remainder_matched.contiguous(), remainder_path)

        meta = {
            "source": "decompose diffmeans_poetry onto refusal_direction",
            "model_id": f"gemma-3-{MODEL_SIZE}-{VARIANT}",
            "matched_norm": target_norm.item(),
            "cos_poetry_refusal": cos_poetry_refusal,
            "cos_sae_poetry_refusal": cos_sae_refusal,
            "proj_fraction_of_poetry_norm": frac,
            "files": {
                "poetry_refusal_parallel.pt": (
                    "refusal-parallel component of poetry, rescaled to |poetry|"
                ),
                "poetry_orthogonal.pt": (
                    "orthogonal 'pure poetry' remainder, rescaled to |poetry|"
                ),
            },
        }
        (PIECES_DIR / "decomposition_metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

        print()
        print(rule)
        print("Saved matched-norm pieces (both rescaled to |poetry| = "
              f"{target_norm.item():.2f})")
        print(rule)
        print(f"  refusal-parallel  → {parallel_path}")
        print(f"  orthogonal pure   → {remainder_path}")


if __name__ == "__main__":
    main()
