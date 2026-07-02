"""Refusal scoring + KL helpers, shared by the refusal and poetry pipelines.

Moved verbatim from ``refusal_directions/select_direction.py``. The only
change is that ``_score_dataset`` now talks to a ``DirectionModel`` adapter
(``model.decoder_layers`` + ``model.last_position_logits``) instead of a
concrete Gemma wrapper, so the same scoring loop runs on any backend.
"""

from __future__ import annotations

import contextlib
import math

import torch

from interpret.sae import HookManager

_REFUSAL_EPSILON = 1e-8
_KL_EPSILON = 1e-6


def _refusal_score(logits: torch.Tensor, refusal_toks: tuple[int, ...]) -> float:
    """Reference score: log P(refusal_tok) - log(1 - P(refusal_tok)).

    Returns ``nan`` if the logits are non-finite (collapsed model) so the
    downstream mean is itself ``nan`` and the filter discards the candidate
    — without this guard a partially-NaN softmax can return a deceptively
    finite refusal-token mass when the argmax token happens to be elsewhere.
    """
    if not torch.isfinite(logits).all():
        return float("nan")
    logits = logits.to(torch.float64)
    probs = torch.softmax(logits, dim=-1)
    refusal_p = probs[list(refusal_toks)].sum().item()
    nonref_p = max(1.0 - refusal_p, _REFUSAL_EPSILON)
    return math.log(refusal_p + _REFUSAL_EPSILON) - math.log(nonref_p)


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) at a single position.

    Returns ``+inf`` if either logit tensor contains non-finite values — this
    happens when the intervention collapses the model (e.g. a strong ablation
    sends a single logit to ``+inf``, which makes softmax produce ``NaN`` at
    that position). Returning ``+inf`` lets the downstream filter discard the
    candidate cleanly instead of poisoning the per-prompt mean with ``NaN``.
    """
    if not (torch.isfinite(p_logits).all() and torch.isfinite(q_logits).all()):
        return float("inf")
    p = torch.softmax(p_logits.to(torch.float64), dim=-1)
    q = torch.softmax(q_logits.to(torch.float64), dim=-1)
    return torch.sum(
        p * (torch.log(p + _KL_EPSILON) - torch.log(q + _KL_EPSILON))
    ).item()


@contextlib.contextmanager
def _maybe_session(manager: HookManager | None, layers):
    """Open a HookManager session if one was provided, else a no-op."""
    if manager is None:
        yield
        return
    with manager.session(layers):
        yield


def _score_dataset(
    model,
    prompts: list[str],
    manager: HookManager | None,
    refusal_toks: tuple[int, ...],
) -> tuple[list[float], list[torch.Tensor]]:
    """Run forward over `prompts` once, returning (refusal_scores, last_logits).

    ``model`` is a ``DirectionModel`` adapter. Any ``manager`` is attached to
    ``model.decoder_layers`` for the duration so steering reaches every
    ``last_position_logits`` forward inside the loop.
    """
    layers = model.decoder_layers
    scores: list[float] = []
    logits_list: list[torch.Tensor] = []
    with _maybe_session(manager, layers):
        for prompt in prompts:
            logits = model.last_position_logits(prompt)
            scores.append(_refusal_score(logits, refusal_toks))
            logits_list.append(logits)
    return scores, logits_list
