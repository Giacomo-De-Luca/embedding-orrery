"""Run a heavy computation so that a cancel request can stop it mid-step.

UMAP, HDBSCAN, PCA, gensim coherence and the SAE weight download spend their
time in compiled code or blocking I/O that a thread cannot interrupt, so a
cooperative check between steps cannot stop them. ``CancellableTask`` runs such
a step in a spawned child process instead; the calling thread polls the job's
cancel event and kills the child (and its own children) when it is set.

The target is a module-level function ``target(channel, *args, **kwargs)``. It
streams intermediate items with ``channel.send(item)`` (progress, or each
finished result) and returns its final value. Items reach ``on_item`` in the
calling thread, which keeps every DuckDB write in the parent: DuckDB allows one
writing process, so a child must never open the database.

Numeric arrays above ``spill_min_bytes`` reach the child through a temporary
``.npy`` file mapped copy-on-write rather than through the pipe, so a large
embedding matrix is neither pickled nor duplicated in memory.

Small inputs are not worth a process start (a second or two of imports), so
callers pass ``isolate=SETTINGS.should_isolate(n_rows)``; inline runs keep the
same API and honour a cancel at each ``channel.send`` and at the end.
Tunables live in ``config/cancellable_tasks.json``.
"""

import contextlib
import multiprocessing
import os
import signal
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..services.cancel_registry import JobCancelled, is_cancelled, raise_if_cancelled
from .json_settings import JsonFileSettings
from .resource_paths import PACKAGE_ROOT


@dataclass(frozen=True)
class CancellableTaskSettings(JsonFileSettings):
    """Tunables for CancellableTask (all optional in the config file)."""

    SETTINGS_LABEL: ClassVar[str] = "cancellable task settings"

    # Inputs with fewer rows run inline: the process start would cost more
    # than the step itself.
    isolate_min_rows: int = 2000
    # How often the calling thread checks the cancel event and the pipe.
    poll_interval_ms: int = 200
    # Numeric arrays at least this large go through a temp file, not the pipe.
    spill_min_bytes: int = 1_048_576
    # How long a terminated child gets to exit before it is killed.
    terminate_grace_ms: int = 3000

    def __post_init__(self):
        self.require_int("isolate_min_rows", minimum=0)
        self.require_int("poll_interval_ms", minimum=1)
        self.require_int("spill_min_bytes", minimum=0)
        self.require_int("terminate_grace_ms", minimum=0)

    def should_isolate(self, n_rows: int) -> bool:
        return n_rows >= self.isolate_min_rows


SETTINGS = CancellableTaskSettings.from_file(PACKAGE_ROOT / "config" / "cancellable_tasks.json")


class TaskFailed(RuntimeError):
    """The target raised in the child process (or the child died)."""

    def __init__(self, error_type: str, message: str, details: str = ""):
        super().__init__(f"{error_type}: {message}" if message else error_type)
        self.error_type = error_type
        self.details = details


@dataclass(frozen=True)
class _SpilledArray:
    path: str


class _InlineChannel:
    def __init__(self, on_item: Callable[[Any], None] | None, cancel_event):
        self._on_item = on_item
        self._cancel_event = cancel_event

    def send(self, item: Any) -> None:
        raise_if_cancelled(self._cancel_event)
        if self._on_item is not None:
            self._on_item(item)


class _PipeChannel:
    def __init__(self, conn):
        self._conn = conn

    def send(self, item: Any) -> None:
        self._conn.send(("item", item))


class CancellableTask:
    """Run one heavy step, killable by the job's cancel event.

    Args:
        cancel_event: The job's cancel event (``None``: not cancellable).
        isolate: Run in a spawned child process (killable mid-step) instead of
            the calling thread.
        on_item: Called in the calling thread with each ``channel.send`` item.
        settings: Defaults to ``config/cancellable_tasks.json``.
    """

    def __init__(
        self,
        cancel_event: threading.Event | None,
        *,
        isolate: bool,
        on_item: Callable[[Any], None] | None = None,
        settings: CancellableTaskSettings | None = None,
    ):
        self.cancel_event = cancel_event
        self.isolate = isolate
        self.on_item = on_item
        self.settings = settings or SETTINGS

    def run(self, target: Callable[..., Any], *args, **kwargs) -> Any:
        """Run ``target(channel, *args, **kwargs)`` and return its result.

        Raises JobCancelled when the job is cancelled before or during the step,
        TaskFailed (isolated) or the target's own exception (inline) on failure.
        """
        raise_if_cancelled(self.cancel_event)
        if not self.isolate:
            result = target(_InlineChannel(self.on_item, self.cancel_event), *args, **kwargs)
            raise_if_cancelled(self.cancel_event)
            return result
        return self._run_isolated(target, args, kwargs)

    def _run_isolated(self, target, args: tuple, kwargs: dict) -> Any:
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="orrery-task-") as tmp:
            spill = _ArraySpill(Path(tmp), self.settings.spill_min_bytes)
            child_args = tuple(spill.spill(a) for a in args)
            child_kwargs = {k: spill.spill(v) for k, v in kwargs.items()}

            receiver, sender = context.Pipe(duplex=False)
            # Not a daemon: targets such as gensim's coherence start worker
            # pools, which daemonic processes may not do.
            process = context.Process(
                target=_child_main,
                args=(sender, os.getpid(), target, child_args, child_kwargs),
                name=f"orrery-task-{getattr(target, '__name__', 'task')}",
            )
            process.start()
            sender.close()
            try:
                result = self._collect(process, receiver)
            except BaseException:  # cancelled, failed or interrupted: stop it now
                self._stop(process, wait=False)
                raise
            finally:
                receiver.close()
            self._stop(process, wait=True)
            return result

    def _collect(self, process, receiver) -> Any:
        poll_seconds = self.settings.poll_interval_ms / 1000
        while True:
            if is_cancelled(self.cancel_event):
                raise JobCancelled()
            if receiver.poll(poll_seconds):
                try:
                    kind, payload = receiver.recv()
                except EOFError:
                    process.join(self.settings.terminate_grace_ms / 1000)
                    raise TaskFailed(
                        "WorkerExit", f"worker exited with code {process.exitcode}"
                    ) from None
                if kind == "item":
                    if self.on_item is not None:
                        self.on_item(payload)
                elif kind == "result":
                    return payload
                else:
                    raise TaskFailed(*payload)
            elif not process.is_alive() and not receiver.poll():
                raise TaskFailed(
                    "WorkerExit",
                    f"worker exited with code {process.exitcode} before returning",
                )

    def _stop(self, process, *, wait: bool) -> None:
        """Reap the child: let a finished one exit (``wait``), else terminate it
        (SIGTERM, SIGKILL after the grace period); then kill anything still
        running in its process group, e.g. worker pools it started."""
        grace = self.settings.terminate_grace_ms / 1000
        pid = process.pid
        if wait:
            process.join(grace)
        if process.is_alive():
            _signal_group(pid, signal.SIGTERM)
            process.join(grace)
        if process.is_alive():
            _signal_group(pid, signal.SIGKILL)
            process.join()
        process.close()
        # The group outlives its leader while members remain, so its id is not
        # reused yet: safe to signal after the child is reaped.
        with contextlib.suppress(OSError, AttributeError):
            os.killpg(pid, signal.SIGKILL)


class _ArraySpill:
    """Write large numeric arrays to temp .npy files; replace them by a path marker."""

    def __init__(self, directory: Path, min_bytes: int):
        self._directory = directory
        self._min_bytes = min_bytes
        self._count = 0

    def spill(self, value: Any) -> Any:
        if (
            isinstance(value, np.ndarray)
            and not value.dtype.hasobject
            and value.nbytes >= self._min_bytes
        ):
            path = self._directory / f"arg{self._count}.npy"
            self._count += 1
            np.save(path, value, allow_pickle=False)
            return _SpilledArray(str(path))
        return value


def _unspill(value: Any) -> Any:
    if isinstance(value, _SpilledArray):
        # Copy-on-write: pages are shared with the file, and libraries that
        # write into their input (numba kernels, in-place sklearn ops) still can.
        return np.asarray(np.load(value.path, mmap_mode="c"))
    return value


def _signal_group(pid: int, signum: int) -> None:
    """Signal the child's process group (it and anything it started)."""
    try:
        os.killpg(pid, signum)
    except (OSError, AttributeError):
        # No group yet (the child had not run setpgrp) or no killpg (Windows).
        with contextlib.suppress(OSError):
            os.kill(pid, signum)


def _watch_parent(parent_pid: int) -> None:
    """Exit if the parent dies, taking the whole group (worker pools included)
    down, so an orphaned step does not keep the CPU busy."""
    while True:
        if os.getppid() != parent_pid:
            with contextlib.suppress(OSError, AttributeError):
                os.killpg(os.getpgrp(), signal.SIGKILL)
            os._exit(1)
        time.sleep(1.0)


def _exit_on_sigterm(signum, frame) -> None:
    """Turn the parent's SIGTERM into SystemExit, so pure-Python steps unwind
    (``with open(...)`` blocks flush and close). A step inside compiled code
    only sees it when control returns to Python; SIGKILL follows after the
    grace period."""
    raise SystemExit(128 + signum)


def _child_main(sender, parent_pid: int, target, args: tuple, kwargs: dict) -> None:
    if hasattr(os, "setpgrp"):
        os.setpgrp()  # own process group: the parent can kill our children too
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    try:
        args = tuple(_unspill(a) for a in args)
        kwargs = {k: _unspill(v) for k, v in kwargs.items()}
        result = target(_PipeChannel(sender), *args, **kwargs)
        sender.send(("result", result))
    except SystemExit:  # terminated by the parent: nobody is listening any more
        raise
    except BaseException as exc:  # report everything; the parent re-raises
        sender.send(("error", (type(exc).__name__, str(exc), traceback.format_exc())))
    finally:
        sender.close()
