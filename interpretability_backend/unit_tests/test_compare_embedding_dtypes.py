"""Agreement statistics of the fp32-vs-bf16 embedding comparison script (no model loads)."""

import numpy as np
import pytest

from interpretability_backend.scripts.compare_embedding_dtypes import (
    EmbeddingAgreement,
    FixedDtypePolicy,
)


def _vectors(n: int = 30, d: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_identical_embeddings_agree_fully():
    vectors = _vectors()

    agreement = EmbeddingAgreement.between(vectors, vectors.copy(), k=5)

    assert agreement.mean_cosine == pytest.approx(1.0)
    assert agreement.min_cosine == pytest.approx(1.0)
    assert agreement.neighbour_overlap == pytest.approx(1.0)


def test_scale_does_not_matter():
    vectors = _vectors()

    agreement = EmbeddingAgreement.between(vectors, vectors * 3.0, k=5)

    assert agreement.min_cosine == pytest.approx(1.0)


def test_small_noise_keeps_high_agreement():
    vectors = _vectors()
    noisy = vectors + np.random.default_rng(1).normal(scale=1e-3, size=vectors.shape)

    agreement = EmbeddingAgreement.between(vectors, noisy.astype(np.float32), k=5)

    assert agreement.min_cosine > 0.999
    assert agreement.neighbour_overlap > 0.9


def test_unrelated_embeddings_disagree():
    agreement = EmbeddingAgreement.between(_vectors(seed=0), _vectors(seed=1), k=5)

    assert agreement.mean_cosine < 0.5
    assert agreement.neighbour_overlap < 0.5


def test_k_is_capped_by_the_number_of_texts():
    vectors = _vectors(n=3)

    agreement = EmbeddingAgreement.between(vectors, vectors, k=10)

    assert agreement.neighbour_overlap == pytest.approx(1.0)


def test_fixed_policy_ignores_device_and_model():
    assert FixedDtypePolicy("bfloat16").for_device("cpu", "any/model") == "bfloat16"
