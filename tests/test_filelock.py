"""Tests for kagura_memory._filelock (cross-process advisory credentials lock)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
