"""
Token streaming event bus for real-time generation output.

Mirrors the architecture of progress_emitter.py but with a different event
shape optimised for per-token streaming. Subscribers register an asyncio.Queue
keyed by stream_id; the sync emit_token() function is thread-safe and can be
called from the model inference thread.
"""

import asyncio
from dataclasses import dataclass


@dataclass
class TokenEvent:
    """A single token from streaming generation."""

    stream_id: str
    token_index: int  # 0-based position in generated output
    token_id: int  # raw SentencePiece token ID
    text: str  # clean text delta for display
    done: bool  # True on the last token
    error: str | None = None


# Global event bus: stream_id -> list of subscriber queues
_token_subscribers: dict[str, list[asyncio.Queue]] = {}
_lock = asyncio.Lock()


async def register_token_subscriber(stream_id: str, queue: asyncio.Queue) -> None:
    """Register a subscriber queue for a generation stream."""
    async with _lock:
        if stream_id not in _token_subscribers:
            _token_subscribers[stream_id] = []
        _token_subscribers[stream_id].append(queue)


async def unregister_token_subscriber(stream_id: str, queue: asyncio.Queue) -> None:
    """Unregister a subscriber queue for a generation stream."""
    async with _lock:
        if stream_id in _token_subscribers:
            try:
                _token_subscribers[stream_id].remove(queue)
                if not _token_subscribers[stream_id]:
                    del _token_subscribers[stream_id]
            except ValueError:
                pass  # Already removed


def emit_token(
    stream_id: str,
    token_index: int,
    token_id: int,
    text: str,
    done: bool,
    error: str | None = None,
) -> None:
    """Emit a token event to all subscribers. Thread-safe (uses put_nowait)."""
    if stream_id not in _token_subscribers:
        return

    event = TokenEvent(
        stream_id=stream_id,
        token_index=token_index,
        token_id=token_id,
        text=text,
        done=done,
        error=error,
    )

    for queue in _token_subscribers.get(stream_id, []):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop if subscriber is slow
