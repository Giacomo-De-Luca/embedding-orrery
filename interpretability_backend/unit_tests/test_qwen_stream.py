"""Unit tests for the Qwen streaming helpers (no model load).

Covers the pieces of ``Qwen3Inference.generate_stream`` that don't need a real
model: the token-id streamer (prompt skip + id collection), the cancel
criterion, and the decode+diff event generator.
"""

import threading

import torch

from interpret.inference.qwen3_transformers import (
    _CancelCriteria,
    _drain_until_sentinel,
    _stream_token_events,
    _TokenIdStreamer,
)


def test_token_id_streamer_skips_prompt_and_collects_tokens():
    streamer = _TokenIdStreamer()
    streamer.put(torch.tensor([[1, 2, 3]]))  # prompt (2-D) -> dropped
    streamer.put(torch.tensor([10]))  # generated token
    streamer.put(torch.tensor([11]))
    streamer.end()
    assert list(_drain_until_sentinel(streamer.queue)) == [10, 11]


def test_cancel_criteria_reflects_event():
    event = threading.Event()
    crit = _CancelCriteria(event)
    assert crit(None, None) is False
    event.set()
    assert crit(None, None) is True


def _fake_decode(token_ids, skip_special_tokens=True):
    # Deterministic id -> text chunk mapping; decode returns the full join so
    # the diff logic recovers the per-token delta.
    charmap = {10: "He", 11: "llo", 12: ",", 13: " world"}
    return "".join(charmap[i] for i in token_ids)


def test_stream_token_events_diff_decode_reconstructs_text():
    events = list(_stream_token_events([10, 11, 12, 13], _fake_decode))

    assert "".join(e.text_delta for e in events) == "Hello, world"
    assert [e.token_id for e in events] == [10, 11, 12, 13]
    assert [e.token_index for e in events] == [0, 1, 2, 3]
    # is_done is True only on the final event.
    assert events[-1].is_done is True
    assert all(e.is_done is False for e in events[:-1])


def test_stream_token_events_empty_source_yields_nothing():
    assert list(_stream_token_events([], _fake_decode)) == []
