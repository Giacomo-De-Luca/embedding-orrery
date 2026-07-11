"""Unit tests for the residual-norm profiler's pure helpers.

CPU-only, no model / SAE download. Feeds synthetic Gemma- and Qwen-shaped
capture dicts and known per-token norms, asserting the adapter, per-token
norm computation (incl. BOS drop), and summary stats.
"""

import torch

from interpret.inference.residual_norm_profiler import (
    _capture_to_layer_tensors,
    compute_token_norms,
    residual_norms_from_capture,
    summarize_layer_norms,
)
from interpret.sae.sae_config import HookType


def _tensor_with_row_norms(norms: list[float]) -> torch.Tensor:
    """Build a [1, seq, 2] tensor whose token rows have the given L2 norms.

    Row i is ``[n_i, 0]`` so ``||row_i|| == n_i``.
    """
    rows = [[n, 0.0] for n in norms]
    return torch.tensor([rows], dtype=torch.float32)  # [1, seq, 2]


class TestComputeTokenNorms:
    def test_norms_over_last_dim(self):
        t = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])  # norms 5, 10
        out = compute_token_norms(t, drop_bos=False)
        assert torch.allclose(out, torch.tensor([5.0, 10.0]))

    def test_drops_bos_position(self):
        t = _tensor_with_row_norms([99.0, 5.0, 10.0])  # BOS sink at pos 0
        out = compute_token_norms(t, drop_bos=True)
        assert torch.allclose(out, torch.tensor([5.0, 10.0]))

    def test_drop_bos_noop_on_single_token(self):
        t = _tensor_with_row_norms([7.0])
        out = compute_token_norms(t, drop_bos=True)
        assert torch.allclose(out, torch.tensor([7.0]))

    def test_accepts_2d_tensor(self):
        t = torch.tensor([[3.0, 4.0], [0.0, 0.0]])  # [seq, d]
        out = compute_token_norms(t, drop_bos=False)
        assert torch.allclose(out, torch.tensor([5.0, 0.0]))

    def test_output_is_fp32_cpu(self):
        t = _tensor_with_row_norms([1.0, 2.0]).to(torch.bfloat16)
        out = compute_token_norms(t, drop_bos=False)
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"


class TestCaptureAdapter:
    def test_gemma_shape(self):
        cache = {
            "prefill": {
                0: {"post_mlp": _tensor_with_row_norms([1.0])},
                5: {"post_mlp": _tensor_with_row_norms([2.0])},
                "final_norm": _tensor_with_row_norms([9.0]),  # must be ignored
            }
        }
        out = _capture_to_layer_tensors(cache)
        assert set(out) == {0, 5}

    def test_qwen_shape_picks_resid_post_only(self):
        cache = {
            0: {HookType.RESID_POST: _tensor_with_row_norms([1.0])},
            5: {
                HookType.RESID_POST: _tensor_with_row_norms([2.0]),
                HookType.MLP_OUT: _tensor_with_row_norms([99.0]),  # ignored
            },
        }
        out = _capture_to_layer_tensors(cache)
        assert set(out) == {0, 5}
        # The layer-5 tensor must be the RESID_POST one (norm 2), not MLP_OUT (99).
        assert torch.allclose(compute_token_norms(out[5], drop_bos=False), torch.tensor([2.0]))

    def test_residual_norms_from_capture_gemma_with_bos_drop(self):
        cache = {"prefill": {3: {"post_mlp": _tensor_with_row_norms([99.0, 4.0, 8.0])}}}
        out = residual_norms_from_capture(cache, drop_bos=True)
        assert set(out) == {3}
        assert torch.allclose(out[3], torch.tensor([4.0, 8.0]))

    def test_residual_norms_from_capture_qwen(self):
        cache = {2: {HookType.RESID_POST: _tensor_with_row_norms([3.0, 6.0])}}
        out = residual_norms_from_capture(cache, drop_bos=False)
        assert torch.allclose(out[2], torch.tensor([3.0, 6.0]))


class TestSummarizeLayerNorms:
    def test_quantiles_and_stats(self):
        per_layer = {0: torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}
        out = summarize_layer_norms(per_layer)
        s = out[0]
        assert s["p25"] == 2.0
        assert s["median"] == 3.0
        assert s["p75"] == 4.0
        assert s["mean"] == 3.0
        assert s["count"] == 5

    def test_sorted_and_drops_empty_layers(self):
        per_layer = {
            5: torch.tensor([2.0, 2.0]),
            0: torch.tensor([1.0, 1.0]),
            9: torch.tensor([]),  # empty -> dropped
        }
        out = summarize_layer_norms(per_layer)
        assert list(out.keys()) == [0, 5]

    def test_returns_plain_floats(self):
        out = summarize_layer_norms({0: torch.tensor([1.0, 3.0])})
        assert all(isinstance(v, float) for k, v in out[0].items() if k != "count")
        assert isinstance(out[0]["count"], int)
