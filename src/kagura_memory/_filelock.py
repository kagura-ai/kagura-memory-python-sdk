"""Cross-process advisory file locking for ``~/.kagura/credentials.json``.

The in-process :class:`asyncio.Lock` in ``auth.credentials`` serializes
credential refreshes *within* a single process. When several processes own
the same credentials file at once — e.g. multiple ``kagura-mcp`` proxy
children (one per Claude Code session) plus a stray ``KaguraClient`` — they
need a *cross-process* gate so a concurrent read-modify-write cannot lose an
update. This module provides that gate as a small advisory-lock context
manager wrapping a lock file next to the target.

POSIX only for v1 (``fcntl.flock``). On platforms without ``fcntl`` (Windows)
the context manager degrades to a **no-op** with a one-time debug log — a
deliberate, documented limitation tracked as a follow-up (the Windows
``msvcrt.locking`` shim has materially different semantics — byte-range,
mandatory, no shared/exclusive distinction — and is out of scope here). The
no-op is safe because the atomic ``os.replace`` write in ``auth.credentials``
already guarantees the file is never torn; the lock only adds *serialization*
of the read-modify-write, which single-owner deployments (the common Claude
Code case) do not require.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger("kagura_memory")


def _lock_path(target: Path) -> Path:
    """Sibling ``.lock`` file for ``target`` (never the credentials file itself).

    Locking a dedicated sibling rather than the credentials file avoids
    interfering with the atomic ``os.replace`` (which swaps the inode out from
    under any fd holding a lock on the original file).
    """
    return target.with_name(target.name + ".lock")


@contextlib.contextmanager
def file_lock(target: Path, *, exclusive: bool = True) -> Iterator[None]:
    """Hold a cross-process advisory lock for ``target`` for the block's body.

    Args:
        target: The file being protected (the lock is taken on a sibling
            ``<name>.lock``, created with mode 0600).
        exclusive: ``True`` for a write lock (``LOCK_EX``), ``False`` for a
            shared read lock (``LOCK_SH``).

    On non-POSIX platforms this is a no-op (see module docstring). The lock is
    always released — including on exception — by the ``finally`` block.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - exercised only on Windows
        logger.debug("fcntl unavailable; credentials file lock is a no-op on this platform")
        yield
        return

    lock_file = _lock_path(target)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    # O_CREAT so the lock file is created on first use; mode 0600 keeps it as
    # private as the credentials file it guards.
    fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
