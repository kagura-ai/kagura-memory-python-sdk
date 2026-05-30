"""Tests for kagura_memory._filelock (cross-process advisory credentials lock)."""

from __future__ import annotations

import builtins
import sys
import time
from pathlib import Path

import pytest

import kagura_memory._filelock as fl
from kagura_memory._filelock import _lock_path, file_lock

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX fcntl lock; Windows is a no-op"
)


def test_lock_path_is_sibling_not_target():
    """The lock is taken on <name>.lock, never the credentials file itself."""
    target = Path("/home/u/.kagura/credentials.json")
    assert _lock_path(target) == Path("/home/u/.kagura/credentials.json.lock")


@_POSIX_ONLY
def test_file_lock_creates_lock_file_and_yields(tmp_path: Path):
    target = tmp_path / "credentials.json"
    with file_lock(target, exclusive=True):
        # The sibling lock file exists while the lock is held.
        assert (tmp_path / "credentials.json.lock").exists()


@_POSIX_ONLY
def test_file_lock_sequential_acquire_release(tmp_path: Path):
    """Re-acquiring after release must not deadlock (lock is released on exit)."""
    target = tmp_path / "credentials.json"
    for _ in range(3):
        with file_lock(target, exclusive=True):
            pass
    with file_lock(target, exclusive=False):  # shared also acquirable afterwards
        pass


@_POSIX_ONLY
def test_file_lock_released_on_exception(tmp_path: Path):
    """An exception inside the block still releases the lock (finally)."""
    target = tmp_path / "credentials.json"
    with pytest.raises(ValueError, match="boom"):
        with file_lock(target, exclusive=True):
            raise ValueError("boom")
    # If the lock leaked, this second acquire would hang; it returns promptly.
    with file_lock(target, exclusive=True):
        pass


@_POSIX_ONLY
def test_file_lock_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "nested" / "credentials.json"
    with file_lock(target, exclusive=True):
        assert target.parent.is_dir()


def test_file_lock_noop_when_fcntl_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """On a platform without fcntl the CM degrades to a no-op (no lock file)."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "fcntl":
            raise ImportError("simulated no fcntl (Windows)")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    target = tmp_path / "credentials.json"
    with file_lock(target, exclusive=True):
        pass
    # No-op path must not have created a lock file.
    assert not (tmp_path / "credentials.json.lock").exists()


# ---------------------------------------------------------------------------
# Windows msvcrt backend (#158)
#
# msvcrt is unavailable on the Linux CI runner, so the backend logic is
# exercised by forcing `import fcntl` to fail and injecting a fake `msvcrt`
# module. This unit-tests the byte-range / no-shared-mode / retry-loop logic
# without a real Windows host.
# ---------------------------------------------------------------------------


class _FakeMsvcrt:
    """Minimal stand-in for the ``msvcrt`` module's locking API."""

    LK_LOCK = 1
    LK_NBLCK = 2
    LK_UNLCK = 0

    def __init__(self, fail_nblck_times: int = 0) -> None:
        self.calls: list[tuple[int, int]] = []
        self._fail = fail_nblck_times

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((mode, nbytes))
        if mode == self.LK_NBLCK and self._fail > 0:
            self._fail -= 1
            raise OSError("Resource temporarily unavailable (simulated contention)")


def _force_msvcrt(monkeypatch: pytest.MonkeyPatch, fake: _FakeMsvcrt) -> None:
    """Make ``_detect_backend`` resolve to ``fake`` (no fcntl, then msvcrt)."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "fcntl":
            raise ImportError("simulated no fcntl (Windows)")
        if name == "msvcrt":
            return fake
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_msvcrt_backend_locks_and_unlocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The msvcrt path creates the lock file and brackets the body lock→unlock."""
    fake = _FakeMsvcrt()
    _force_msvcrt(monkeypatch, fake)

    target = tmp_path / "credentials.json"
    with file_lock(target, exclusive=True):
        assert (tmp_path / "credentials.json.lock").exists()

    modes = [mode for mode, _ in fake.calls]
    assert modes == [fake.LK_NBLCK, fake.LK_UNLCK]
    assert all(nbytes == 1 for _, nbytes in fake.calls)


def test_msvcrt_backend_treats_shared_as_exclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Windows has no shared mode: exclusive=False still takes an exclusive lock."""
    fake = _FakeMsvcrt()
    _force_msvcrt(monkeypatch, fake)

    target = tmp_path / "credentials.json"
    with file_lock(target, exclusive=False):
        pass

    # Same NBLCK acquire as the exclusive case — no shared variant exists.
    assert fake.calls[0][0] == fake.LK_NBLCK


def test_msvcrt_backend_retries_on_contention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A busy lock (LK_NBLCK raising OSError) is retried until it succeeds."""
    fake = _FakeMsvcrt(fail_nblck_times=2)
    _force_msvcrt(monkeypatch, fake)
    monkeypatch.setattr(fl, "_MSVCRT_RETRY_SEC", 0)  # don't actually sleep

    target = tmp_path / "credentials.json"
    with file_lock(target, exclusive=True):
        pass

    # 2 failed NBLCK attempts + 1 success + 1 UNLCK release.
    nblck = [m for m, _ in fake.calls if m == fake.LK_NBLCK]
    assert len(nblck) == 3
    assert fake.calls[-1][0] == fake.LK_UNLCK


# ---------------------------------------------------------------------------
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


@_POSIX_ONLY
def test_file_lock_serializes_across_processes(tmp_path: Path):
    """N processes doing read-modify-write under the lock lose zero updates.

    A deterministic count==N*iters proves cross-process mutual exclusion. The
    matching negative control (no lock → lost updates) is the deterministic
    in-process ``test_refresh_dedup_negative_control_without_lock`` in
    tests/test_auth_credentials.py — a subprocess negative control would be
    inherently racy, which we avoid (flaky tests erode trust).
    """
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    cred = tmp_path / "credentials.json"
    counter = tmp_path / "counter.txt"
    counter.write_text("0")

    n_procs, iters = 4, 5
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
