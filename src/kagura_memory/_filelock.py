"""Cross-process advisory file locking for ``~/.kagura/credentials.json``.

The in-process :class:`asyncio.Lock` in ``auth.credentials`` serializes
credential refreshes *within* a single process. When several processes own
the same credentials file at once — e.g. multiple ``kagura-mcp`` proxy
children (one per Claude Code session) plus a stray ``KaguraClient`` — they
need a *cross-process* gate so a concurrent read-modify-write cannot lose an
update. This module provides that gate as a small advisory-lock context
manager wrapping a lock file next to the target.

Two backends, selected at runtime by which module imports:

- **POSIX** (``fcntl.flock``): whole-file advisory lock, with a real
  shared/exclusive distinction (``LOCK_SH`` / ``LOCK_EX``).
- **Windows** (``msvcrt.locking``): byte-range, *mandatory* lock. The Windows
  API has **no shared-lock mode**, so :func:`file_lock` treats
  ``exclusive=False`` as exclusive too — concurrent readers serialize on
  Windows where they would run in parallel on POSIX. This is acceptable for
  the credentials file (low contention, short critical section) and is the
  documented semantic difference between the backends. ``msvcrt.locking``
  also has no blocking-until-acquired mode that waits indefinitely, so the
  Windows path uses a non-blocking lock attempt (``LK_NBLCK``) in a short
  retry loop to emulate ``flock``'s blocking acquire.

On a platform that has neither module the context manager degrades to a
**no-op** with a one-time debug log. The no-op is safe because the atomic
``os.replace`` write in ``auth.credentials`` already guarantees the file is
never torn; the lock only adds *serialization* of the read-modify-write,
which single-owner deployments (the common Claude Code case) do not require.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger("kagura_memory")

# msvcrt locks a byte range from the current file position; one byte at offset
# 0 is enough to gate the whole sibling lock file (Windows permits locking a
# region at/beyond EOF, so the empty lock file needs no pre-write).
_MSVCRT_LOCK_BYTES = 1
# Poll interval while waiting for a contended msvcrt lock (LK_NBLCK retry loop).
_MSVCRT_RETRY_SEC = 0.05


def _lock_path(target: Path) -> Path:
    """Sibling ``.lock`` file for ``target`` (never the credentials file itself).

    Locking a dedicated sibling rather than the credentials file avoids
    interfering with the atomic ``os.replace`` (which swaps the inode out from
    under any fd holding a lock on the original file).
    """
    return target.with_name(target.name + ".lock")


def _detect_backend() -> tuple[str, Any]:
    """Return ``(name, module)`` for the available lock backend.

    ``name`` is ``"fcntl"``, ``"msvcrt"``, or ``"noop"`` (module is ``None``
    for ``"noop"``). POSIX ``fcntl`` is preferred when both somehow resolve.
    """
    try:
        import fcntl

        return "fcntl", fcntl
    except ImportError:
        pass
    try:
        import msvcrt

        return "msvcrt", msvcrt
    except ImportError:  # pragma: no cover - only on exotic platforms (no fcntl, no msvcrt)
        return "noop", None


def _lock_fd(backend: str, mod: Any, fd: int, *, exclusive: bool) -> None:
    """Acquire the advisory lock on ``fd`` using the selected backend."""
    if backend == "fcntl":
        mod.flock(fd, mod.LOCK_EX if exclusive else mod.LOCK_SH)
        return
    # msvcrt: no shared mode (exclusive param is honored for API parity but
    # every lock is exclusive). LK_NBLCK never blocks; loop to emulate
    # flock's blocking acquire without msvcrt's 10-second LK_LOCK timeout.
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        try:
            mod.locking(fd, mod.LK_NBLCK, _MSVCRT_LOCK_BYTES)
            return
        except OSError:
            time.sleep(_MSVCRT_RETRY_SEC)


def _unlock_fd(backend: str, mod: Any, fd: int) -> None:
    """Release the advisory lock on ``fd`` using the selected backend."""
    if backend == "fcntl":
        mod.flock(fd, mod.LOCK_UN)
        return
    os.lseek(fd, 0, os.SEEK_SET)
    mod.locking(fd, mod.LK_UNLCK, _MSVCRT_LOCK_BYTES)


@contextlib.contextmanager
def file_lock(target: Path, *, exclusive: bool = True) -> Iterator[None]:
    """Hold a cross-process advisory lock for ``target`` for the block's body.

    Args:
        target: The file being protected (the lock is taken on a sibling
            ``<name>.lock``, created with mode 0600).
        exclusive: ``True`` for a write lock, ``False`` for a shared read lock.
            On the Windows (``msvcrt``) backend there is no shared mode, so
            ``False`` is upgraded to an exclusive lock (see module docstring).

    On a platform with neither ``fcntl`` nor ``msvcrt`` this is a no-op. The
    lock is always released — including on exception — by the ``finally`` block.
    """
    backend, mod = _detect_backend()
    if backend == "noop":  # pragma: no cover - only on exotic platforms
        logger.debug("no fcntl/msvcrt; credentials file lock is a no-op on this platform")
        yield
        return

    lock_file = _lock_path(target)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    # O_CREAT so the lock file is created on first use; mode 0600 keeps it as
    # private as the credentials file it guards.
    fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        _lock_fd(backend, mod, fd, exclusive=exclusive)
        try:
            yield
        finally:
            _unlock_fd(backend, mod, fd)
    finally:
        os.close(fd)
