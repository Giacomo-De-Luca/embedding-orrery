"""Module-level task targets for test_cancellable_task.py.

A spawned child imports its target by module path, so test targets must live
in an importable module rather than inside the test file's functions.
"""

import os
import subprocess
import time
from pathlib import Path

import numpy as np


def stream_squares(channel, n: int) -> str:
    """Send n items, then return a summary."""
    for i in range(n):
        channel.send(i * i)
    return f"sent {n}"


def sum_rows(channel, matrix: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Exercise array spilling: the child receives the (possibly memory-mapped) array."""
    channel.send(("shape", matrix.shape, matrix.dtype.str, matrix.flags.writeable))
    return matrix.sum(axis=1) * scale


def fail(channel, message: str) -> None:
    raise ValueError(message)


def sleep_forever(channel, marker: str) -> None:
    """Write our pid to ``marker``, report it, then block until killed."""
    Path(marker).write_text(str(os.getpid()))
    channel.send("started")
    while True:
        time.sleep(0.05)


def exit_hard(channel, code: int) -> None:
    os._exit(code)


def spawn_grandchild_and_block(channel, marker: str) -> None:
    """Start a long-lived grandchild (like a worker pool would), report its pid, block."""
    grandchild = subprocess.Popen(["sleep", "60"])
    Path(marker).write_text(str(grandchild.pid))
    channel.send("started")
    while True:
        time.sleep(0.05)
