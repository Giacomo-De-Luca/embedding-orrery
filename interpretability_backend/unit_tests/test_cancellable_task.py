"""CancellableTask: heavy steps that a cancel request can stop mid-computation.

Isolated mode runs the target in a spawned child (killable mid-step); inline
mode runs it in the calling thread and honours a cancel at each channel send.
Both stream channel items to ``on_item`` in the calling thread and return the
target's result.
"""

import os
import threading
import time

import numpy as np
import pytest

from backend.services.cancel_registry import JobCancelled
from backend.utils.cancellable_task import (
    CancellableTask,
    CancellableTaskSettings,
    TaskFailed,
)
from unit_tests import cancellable_task_workers as workers

SPILL_ALL = CancellableTaskSettings(spill_min_bytes=0, poll_interval_ms=20)


@pytest.fixture(params=[True, False], ids=["isolated", "inline"])
def isolate(request):
    return request.param


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestResultsAndItems:
    def test_streams_items_and_returns_result(self, isolate):
        items = []
        task = CancellableTask(None, isolate=isolate, on_item=items.append, settings=SPILL_ALL)

        assert task.run(workers.stream_squares, 4) == "sent 4"
        assert items == [0, 1, 4, 9]

    def test_large_arrays_reach_the_child(self, isolate):
        items = []
        matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
        task = CancellableTask(None, isolate=isolate, on_item=items.append, settings=SPILL_ALL)

        result = task.run(workers.sum_rows, matrix, scale=2.0)

        np.testing.assert_allclose(result, matrix.sum(axis=1) * 2.0)
        _, shape, dtype, writeable = items[0]
        assert shape == (3, 4) and np.dtype(dtype) == np.float32
        # Copy-on-write mapping: libraries that write in place still work.
        assert writeable

    def test_target_error_surfaces_with_its_message(self, isolate):
        task = CancellableTask(None, isolate=isolate, settings=SPILL_ALL)

        with pytest.raises((TaskFailed, ValueError), match="bad input"):
            task.run(workers.fail, "bad input")

    def test_child_exit_without_result_is_a_failure(self):
        task = CancellableTask(None, isolate=True, settings=SPILL_ALL)

        with pytest.raises(TaskFailed, match="exited with code 3"):
            task.run(workers.exit_hard, 3)


class TestCancellation:
    def test_cancel_before_start_never_runs(self, isolate):
        event = threading.Event()
        event.set()
        items = []
        task = CancellableTask(event, isolate=isolate, on_item=items.append, settings=SPILL_ALL)

        with pytest.raises(JobCancelled):
            task.run(workers.stream_squares, 3)
        assert items == []

    def test_inline_cancel_stops_at_next_send(self):
        event = threading.Event()
        items = []

        def on_item(item):
            items.append(item)
            event.set()

        task = CancellableTask(event, isolate=False, on_item=on_item, settings=SPILL_ALL)

        with pytest.raises(JobCancelled):
            task.run(workers.stream_squares, 5)
        assert items == [0]

    def test_isolated_cancel_kills_a_busy_child(self, tmp_path):
        """The child is blocked in compiled-style code; only killing it stops it."""
        event = threading.Event()
        marker = tmp_path / "pid"

        def on_item(item):
            assert item == "started"
            event.set()  # the user clicks Cancel while the step runs

        task = CancellableTask(event, isolate=True, on_item=on_item, settings=SPILL_ALL)

        start = time.monotonic()
        with pytest.raises(JobCancelled):
            task.run(workers.sleep_forever, str(marker))
        assert time.monotonic() - start < 10

        pid = int(marker.read_text())
        deadline = time.monotonic() + 5
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(pid)

    def test_isolated_cancel_also_kills_grandchildren(self, tmp_path):
        """Worker pools a step starts (joblib, gensim) die with it."""
        event = threading.Event()
        marker = tmp_path / "grandchild"
        task = CancellableTask(
            event, isolate=True, on_item=lambda item: event.set(), settings=SPILL_ALL
        )

        with pytest.raises(JobCancelled):
            task.run(workers.spawn_grandchild_and_block, str(marker))

        pid = int(marker.read_text())
        deadline = time.monotonic() + 5
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(pid)


class TestSettings:
    def test_defaults_isolate_large_inputs_only(self):
        settings = CancellableTaskSettings()
        assert settings.should_isolate(settings.isolate_min_rows)
        assert not settings.should_isolate(settings.isolate_min_rows - 1)

    def test_rejects_invalid_values(self):
        with pytest.raises(ValueError):
            CancellableTaskSettings(poll_interval_ms=0)
        with pytest.raises(ValueError):
            CancellableTaskSettings(isolate_min_rows=-1)
