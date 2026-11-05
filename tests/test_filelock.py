"""Tests for kagura_memory._filelock (cross-process advisory credentials lock)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from kagura_memory._filelock import _lock_path, async_file_lock, file_lock
from kagura_memory.exceptions import KaguraAuthError

# Module path for monkeypatching module-level lock constants (string-target form
# so we don't also `import ... as fl`, keeping a single import style).
_FL = "kagura_memory._filelock"


def test_lock_path_is_sibling_not_target():
    """The lock is taken on <name>.lock, never the credentials file itself."""
    target = Path("/home/u/.kagura/credentials.json")
    assert _lock_path(target) == Path("/home/u/.kagura/credentials.json.lock-v1")


def test_file_lock_creates_lock_file_and_yields(tmp_path: Path):
    target = tmp_path / "credentials.json"
    with file_lock(target, exclusive=True):
        # The sibling lock file exists while the lock is held.
        assert (tmp_path / "credentials.json.lock-v1").exists()


def test_file_lock_sequential_acquire_release(tmp_path: Path):
    """Re-acquiring after release must not deadlock (lock is released on exit)."""
    target = tmp_path / "credentials.json"
    for _ in range(3):
        with file_lock(target, exclusive=True):
            pass
    with file_lock(target, exclusive=False):  # shared also acquirable afterwards
        pass


def test_file_lock_released_on_exception(tmp_path: Path):
    """An exception inside the block still releases the lock (finally)."""
    target = tmp_path / "credentials.json"
    with pytest.raises(ValueError, match="boom"):
        with file_lock(target, exclusive=True):
            raise ValueError("boom")
    # If the lock leaked, this second acquire would hang; it returns promptly.
    with file_lock(target, exclusive=True):
        pass


def test_file_lock_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "nested" / "credentials.json"
    with file_lock(target, exclusive=True):
        assert target.parent.is_dir()


def test_legacy_marker_requires_manual_migration(tmp_path):
    target = tmp_path / "credentials.json"
    legacy = tmp_path / "credentials.json.lock"
    legacy.write_text("legacy")
    with pytest.raises(KaguraAuthError, match="Stop every old and new client"):
        with file_lock(target):
            pytest.fail("entered despite legacy marker")
    assert legacy.read_text() == "legacy"


def test_abandoned_marker_is_not_broken(tmp_path, monkeypatch):
    target = tmp_path / "credentials.json"
    marker = _lock_path(target)
    marker.write_text("crashed")
    os.utime(marker, (0, 0))
    monkeypatch.setattr(f"{_FL}._LOCK_ACQUIRE_TIMEOUT_SEC", 0)
    with pytest.raises(TimeoutError, match="stop every participating client"):
        with file_lock(target):
            pytest.fail("broke an old lock")
    assert marker.read_text() == "crashed"


def test_replacement_owner_survives_release(tmp_path):
    target = tmp_path / "credentials.json"
    with file_lock(target):
        marker = _lock_path(target)
        assert json.loads(marker.read_bytes())["protocol"] == "kagura-credentials-lock-v1"
        marker.write_text("replacement")
    assert marker.read_text() == "replacement"


# Cross-process lock contention (subprocess-based, #158)
# ---------------------------------------------------------------------------


def _contend_worker(cred_path: str, counter_path: str, barrier, iters: int) -> None:
    """Increment a shared counter under the file lock; lost updates fail the test."""
    from pathlib import Path as _Path

    from kagura_memory._filelock import file_lock as _file_lock

    cred = _Path(cred_path)
    counter = _Path(counter_path)
    barrier.wait()  # release all workers simultaneously to maximize contention
    for _ in range(iters):
        with _file_lock(cred, exclusive=True):
            value = int(counter.read_text() or "0")
            time.sleep(0.001)  # widen the read-modify-write window
            counter.write_text(str(value + 1))


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="uses mp 'fork' start method; fork is unsafe/non-default on macOS, absent on Windows",
)
def test_file_lock_serializes_across_processes(tmp_path: Path):
    """N processes doing read-modify-write under the lock lose zero updates.

    A deterministic count==N*iters proves cross-process mutual exclusion. The
    contention (high iteration count + a held read-modify-write window) is sized
    so that WITHOUT the lock — e.g. if file_lock regressed to the no-op backend —
    lost updates are near-certain and the count would fall short. The matching
    deterministic negative control (no lock → both refresh) is the in-process
    ``test_refresh_dedup_negative_control_without_lock`` in
    tests/test_auth_credentials.py; a subprocess negative control would be
    inherently racy, which we avoid (flaky tests erode trust).
    """
    import multiprocessing as mp

    # 'fork' lets the worker inherit the Barrier (a spawn context cannot pickle
    # it as a plain arg). Gated to Linux above where fork is the safe default.
    ctx = mp.get_context("fork")
    cred = tmp_path / "credentials.json"
    counter = tmp_path / "counter.txt"
    counter.write_text("0")

    n_procs, iters = 4, 25
    barrier = ctx.Barrier(n_procs)
    procs = [
        ctx.Process(target=_contend_worker, args=(str(cred), str(counter), barrier, iters))
        for _ in range(n_procs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)  # hard timeout: a deadlock must fail, not hang CI
        if p.is_alive():
            p.terminate()
            p.join()
            pytest.fail("lock-contention worker timed out — possible cross-process deadlock")
        assert p.exitcode == 0

    assert int(counter.read_text()) == n_procs * iters


# ---------------------------------------------------------------------------
# async_file_lock — event-loop-friendly, cancellation-safe (#158)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_file_lock_acquires_and_releases(tmp_path: Path):
    """The async lock creates the lock file and releases so it can be re-taken."""
    target = tmp_path / "credentials.json"
    async with async_file_lock(target, exclusive=True):
        assert (tmp_path / "credentials.json.lock-v1").exists()
    # A second acquire after release must not hang.
    async with async_file_lock(target, exclusive=True):
        pass


@pytest.mark.asyncio
async def test_async_file_lock_waits_and_is_cancellation_safe(tmp_path: Path):
    """A waiter blocked on a held lock can be cancelled WITHOUT leaking the lock.

    This is the failure mode the async lock exists to prevent: offloading a
    blocking ``__enter__`` to a worker thread would let the thread win the lock
    after the awaiting task is cancelled, leaking it forever. Here the waiter
    only ever awaits ``asyncio.sleep`` while not holding the lock, so cancelling
    it holds nothing.
    """
    target = tmp_path / "credentials.json"
    release = asyncio.Event()
    acquired_b = False

    async def holder() -> None:
        async with async_file_lock(target, exclusive=True):
            await release.wait()

    async def waiter() -> None:
        nonlocal acquired_b
        async with async_file_lock(target, exclusive=True):
            acquired_b = True

    h = asyncio.create_task(holder())
    await asyncio.sleep(0.05)  # let the holder acquire first
    w = asyncio.create_task(waiter())
    await asyncio.sleep(0.1)  # waiter polls but cannot acquire
    assert not acquired_b, "waiter must not acquire while the holder holds the lock"

    w.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w

    # If the cancelled waiter had leaked the lock, this fresh acquire would hang
    # (and the join-free test would time out). Release the holder, then prove
    # the lock is cleanly re-acquirable.
    release.set()
    await h
    async with async_file_lock(target, exclusive=True):
        pass


@pytest.mark.asyncio
async def test_async_file_lock_times_out_on_held_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A contended async acquire raises TimeoutError once the deadline passes."""
    monkeypatch.setattr(f"{_FL}._LOCK_RETRY_SEC", 0)
    monkeypatch.setattr(f"{_FL}._LOCK_ACQUIRE_TIMEOUT_SEC", 0.0)  # trip immediately

    target = tmp_path / "credentials.json"
    async with async_file_lock(target, exclusive=True):
        # A second acquire on a different fd of the same already-held lock.
        with pytest.raises(TimeoutError):
            async with async_file_lock(target, exclusive=True):
                pass
