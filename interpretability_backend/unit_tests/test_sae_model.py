"""Tests pinning the SAE encode/decode math to the official conventions.

Gemma-scope JumpReLU SAEs are trained with ``apply_b_dec_to_input=False``
(SAELens loader config; DeepMind reference snippet encodes the raw
input), so ``b_dec`` must participate only in ``decode``. A regression
here silently skews every downstream activation (prompt activations,
``sae_document_activations``, SAE-pooled probing features).

Small float32 tensors, no weight downloads.
"""

import torch

from interpret.sae.sae_model import JumpReLUSAE, TopKSAE

D_IN = 8
D_SAE = 16


def _make_jumprelu(seed: int = 0) -> JumpReLUSAE:
    gen = torch.Generator().manual_seed(seed)
    sae = JumpReLUSAE(d_in=D_IN, d_sae=D_SAE)
    sae.w_enc.data = torch.randn(D_IN, D_SAE, generator=gen)
    sae.w_dec.data = torch.randn(D_SAE, D_IN, generator=gen)
    sae.b_enc.data = torch.randn(D_SAE, generator=gen)
    sae.b_dec.data = torch.randn(D_IN, generator=gen)
    # Gemma-scope thresholds are positive (exp of a learned log-threshold).
    sae.threshold.data = torch.rand(D_SAE, generator=gen) * 0.5 + 0.1
    return sae


class TestJumpReLUEncode:
    def test_matches_official_reference(self):
        """encode == relu(x @ W_enc + b_enc) masked by threshold — the
        DeepMind Gemma-scope reference, with no b_dec on the input."""
        sae = _make_jumprelu()
        x = torch.randn(5, D_IN, generator=torch.Generator().manual_seed(1))

        ref_pre = x @ sae.w_enc + sae.b_enc
        ref = torch.relu(ref_pre) * (ref_pre > sae.threshold)

        torch.testing.assert_close(sae.encode(x), ref)

    def test_encode_ignores_b_dec(self):
        sae = _make_jumprelu()
        x = torch.randn(5, D_IN, generator=torch.Generator().manual_seed(2))
        acts = sae.encode(x)

        sae.b_dec.data = torch.full((D_IN,), 100.0)

        torch.testing.assert_close(sae.encode(x), acts)

    def test_decode_adds_b_dec(self):
        sae = _make_jumprelu()
        feats = torch.rand(3, D_SAE, generator=torch.Generator().manual_seed(3))

        torch.testing.assert_close(sae.decode(feats), feats @ sae.w_dec + sae.b_dec)

    def test_threshold_gates_activations(self):
        sae = _make_jumprelu()
        sae.w_enc.data = torch.eye(D_IN, D_SAE)
        sae.b_enc.data = torch.zeros(D_SAE)
        sae.threshold.data = torch.full((D_SAE,), 0.5)

        x = torch.zeros(1, D_IN)
        x[0, 0] = 0.4  # below threshold -> zeroed
        x[0, 1] = 0.5  # at threshold (strict >) -> zeroed
        x[0, 2] = 0.9  # above threshold -> passes through unchanged

        acts = sae.encode(x)
        assert acts[0, 0] == 0.0
        assert acts[0, 1] == 0.0
        torch.testing.assert_close(acts[0, 2], torch.tensor(0.9))


class TestTopKEncode:
    def test_encode_ignores_b_dec_and_keeps_raw_topk(self):
        gen = torch.Generator().manual_seed(4)
        k = 3
        sae = TopKSAE(d_in=D_IN, d_sae=D_SAE, k=k)
        sae.w_enc.data = torch.randn(D_IN, D_SAE, generator=gen)
        sae.b_enc.data = torch.randn(D_SAE, generator=gen)
        sae.b_dec.data = torch.randn(D_IN, generator=gen)

        x = torch.randn(4, D_IN, generator=gen)
        acts = sae.encode(x)

        assert (acts != 0).sum(dim=-1).eq(k).all()
        pre = x @ sae.w_enc + sae.b_enc
        kept = acts != 0
        torch.testing.assert_close(acts[kept], pre[kept])

        sae.b_dec.data = torch.full((D_IN,), 100.0)
        torch.testing.assert_close(sae.encode(x), acts)
