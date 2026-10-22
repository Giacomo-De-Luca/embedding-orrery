"""Batch size, input-length cap and truncation counting in the QWEN and BGE embedders.

QWEN used to run every document it was given in one forward pass (its
``batch_size`` was never read); BGE ran on MPS whatever the device. Both now
honour the job's batch size and cap, and count the documents they truncate.
Model loading is faked: transformers' loaders are patched, FlagEmbedding is a
stub module.
"""

import sys
import types

import numpy as np
import pytest
import torch

from backend.embedding_functions.specific_functions import embed_qwen

BGE_MODULE = "backend.embedding_functions.specific_functions.embed_bge"


class _Batch(dict):
    def to(self, device):
        return self


class FakeQwenTokenizer:
    """Whitespace tokenizer that left-pads like Qwen's (padding_side="left")."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, texts, padding=False, truncation=False, max_length=None, **kwargs):
        lengths = [len(text.split()) for text in texts]
        self.calls.append({"n": len(texts), "truncation": truncation, "max_length": max_length})
        if kwargs.get("return_tensors") is None:
            return {"input_ids": [[1] * n for n in lengths]}
        if truncation:
            lengths = [min(n, max_length) for n in lengths]
        width = max(lengths)
        mask = torch.tensor([[0] * (width - n) + [1] * n for n in lengths])
        return _Batch(input_ids=mask.clone(), attention_mask=mask)


class FakeQwenModel:
    device = "cpu"

    def __init__(self):
        self.batch_sizes: list[int] = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        self.batch_sizes.append(input_ids.shape[0])
        # Last-token state = number of real tokens, so outputs reveal row order.
        hidden = attention_mask.sum(dim=1, keepdim=True).float().repeat(1, 3)
        states = hidden.unsqueeze(1).repeat(1, input_ids.shape[1], 1)
        return types.SimpleNamespace(last_hidden_state=states)


@pytest.fixture
def qwen(monkeypatch):
    tokenizer, model = FakeQwenTokenizer(), FakeQwenModel()
    monkeypatch.setattr(
        embed_qwen.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer, raising=False
    )
    monkeypatch.setattr(
        embed_qwen.AutoModel, "from_pretrained", lambda *a, **k: model, raising=False
    )
    return tokenizer, model


class TestQwen:
    def test_documents_are_embedded_in_batches_of_batch_size(self, qwen):
        _, model = qwen
        ef = embed_qwen.EmbedTextQWEN(model="q", device="cpu", batch_size=2, max_length=10)

        vectors = ef(["a", "a b", "a b c", "a b c d", "a b c d e"])

        assert model.batch_sizes == [2, 2, 1]
        assert [float(v[0]) for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert vectors[0].dtype == np.float32

    def test_cap_truncates_and_counts(self, qwen):
        tokenizer, _ = qwen
        ef = embed_qwen.EmbedTextQWEN(model="q", device="cpu", batch_size=8, max_length=3)

        vectors = ef(["a", "a b c d", "a b", "a b c d e"])

        forward_calls = [c for c in tokenizer.calls if c["truncation"]]
        assert forward_calls and all(c["max_length"] == 3 for c in forward_calls)
        assert [float(v[0]) for v in vectors] == [1.0, 3.0, 2.0, 3.0]
        assert (ef.truncation.truncated, ef.truncation.seen) == (2, 4)


class FakeBGEModel:
    instances: list["FakeBGEModel"] = []

    # Same keyword names as FlagEmbedding's M3Embedder (`devices`, not `device`).
    def __init__(self, model, devices=None, pooling_method=None, use_fp16=None):
        self.devices = devices
        self.tokenizer = FakeQwenTokenizer()
        self.encode_calls: list[dict] = []
        FakeBGEModel.instances.append(self)

    def encode(self, input, batch_size=None, max_length=None):
        self.encode_calls.append({"batch_size": batch_size, "max_length": max_length})
        return {"dense_vecs": np.zeros((len(input), 4), dtype=np.float32)}


@pytest.fixture
def bge_module(monkeypatch):
    fake_flag = types.ModuleType("FlagEmbedding")
    fake_flag.BGEM3FlagModel = FakeBGEModel
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_flag)
    monkeypatch.delitem(sys.modules, BGE_MODULE, raising=False)
    FakeBGEModel.instances = []
    __import__(BGE_MODULE)
    return sys.modules[BGE_MODULE]


class TestBge:
    def test_device_batch_size_and_cap_reach_the_model(self, bge_module):
        ef = bge_module.EmbedTextBGE(model="b", device="cpu", batch_size=5, max_length=64)

        vectors = ef(["a b", "c"])

        model = FakeBGEModel.instances[-1]
        assert model.devices == "cpu"
        assert model.encode_calls == [{"batch_size": 5, "max_length": 64}]
        assert len(vectors) == 2

    def test_documents_over_the_cap_are_counted(self, bge_module):
        ef = bge_module.EmbedTextBGE(model="b", device="cpu", batch_size=5, max_length=2)

        ef(["a b", "a b c", "a"])

        assert (ef.truncation.truncated, ef.truncation.seen) == (1, 3)
