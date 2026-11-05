"""Portable exclusive-create credential lock shared with the TypeScript SDK."""

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from uuid import uuid4

from .exceptions import KaguraAuthError

_LOCK_RETRY_SEC = 0.05
_LOCK_ACQUIRE_TIMEOUT_SEC = 90.0


def _lock_path(target: Path) -> Path:
    return target.with_name(target.name + ".lock-v1")


def _refuse_legacy(target: Path) -> None:
    legacy = target.with_name(target.name + ".lock")
    try:
        legacy.lstat()
    except FileNotFoundError:
        return
    raise KaguraAuthError(
        f"Legacy credentials lock at {legacy}. Stop every old and new client before "
        "manually retiring this marker. Do not restart mixed versions against the same file."
    )


def _try_acquire(target: Path, deadline: float) -> bytes | None:
    _refuse_legacy(target)
    lock = _lock_path(target)
    owner = json.dumps(
        {
            "protocol": "kagura-credentials-lock-v1",
            "owner": str(uuid4()),
            "pid": os.getpid(),
        }
    ).encode()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
    except FileExistsError:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out acquiring {lock}. If abandoned, stop every participating client "
                "before removing it. Age alone cannot establish abandonment."
            ) from None
        return None
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(owner)
        _refuse_legacy(target)
    except BaseException:
        lock.unlink()
        raise
    return owner


def _release(target: Path, owner: bytes) -> None:
    lock = _lock_path(target)
    try:
        if lock.read_bytes() == owner:
            lock.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def file_lock(target: Path, *, exclusive: bool = True) -> Iterator[None]:
    """Acquire the exclusive lock (also for callers requesting a shared lock)."""
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_ACQUIRE_TIMEOUT_SEC
    while (owner := _try_acquire(target, deadline)) is None:
        time.sleep(_LOCK_RETRY_SEC)
    try:
        yield
    finally:
        _release(target, owner)


@asynccontextmanager
async def async_file_lock(target: Path, *, exclusive: bool = True) -> AsyncIterator[None]:
    """Poll without blocking the event loop or leaving a waiter after cancellation."""
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = time.monotonic() + _LOCK_ACQUIRE_TIMEOUT_SEC
    while (owner := _try_acquire(target, deadline)) is None:
        await asyncio.sleep(_LOCK_RETRY_SEC)
    try:
        yield
    finally:
        _release(target, owner)
