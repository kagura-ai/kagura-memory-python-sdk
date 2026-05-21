"""Tests for the ``_exc_message`` helper and the wrappers that consume it.

Originating bug (#127): ``raise click.ClickException(f"Setup failed: {e}")``
rendered ``"Setup failed:"`` when ``str(e) == ""`` (any ``Exception`` raised
with no args, e.g. ``raise RuntimeError()``). The follow-up (#130) extracts
the inline defensive expression into ``_exc_message`` and applies it to every
``ClickException`` / ``KaguraConnectionError`` wrapper across the SDK.

These tests pin the rendering invariant: for any falsy ``str(e)``, the helper
yields the class name so prefixed messages never strand a dangling colon.
"""

from __future__ import annotations

from collections.abc import Callable

import click
import httpx
import pytest

from kagura_memory.exceptions import (
    KaguraAuthDeniedError,
    KaguraAuthError,
    KaguraConnectionError,
    KaguraError,
    _exc_message,
)


class _CustomKaguraError(KaguraError):
    """Test subclass to exercise inheritance handling."""


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected",
    [
        # Empty-arg exceptions fall back to the class name
        (RuntimeError(), "RuntimeError"),
        (ValueError(), "ValueError"),
        (Exception(), "Exception"),
        (KaguraError(), "KaguraError"),
        (KaguraAuthError(), "KaguraAuthError"),
        (KaguraConnectionError(), "KaguraConnectionError"),
        (KaguraAuthDeniedError(), "KaguraAuthDeniedError"),
        (_CustomKaguraError(), "_CustomKaguraError"),
        # Empty-string message is also falsy
        (RuntimeError(""), "RuntimeError"),
        # Non-empty messages pass through
        (RuntimeError("boom"), "boom"),
        (KaguraConnectionError("HTTP 503"), "HTTP 503"),
        # Whitespace is truthy and preserved
        (RuntimeError(" "), " "),
        # BaseException subclasses (not just Exception) are accepted
        (KeyboardInterrupt(), "KeyboardInterrupt"),
        (SystemExit(), "SystemExit"),
    ],
)
def test_exc_message_returns_message_or_class_name(exc: BaseException, expected: str) -> None:
    """``_exc_message(e)`` returns ``str(e)`` when non-empty, else ``e.__class__.__name__``."""
    assert _exc_message(exc) == expected


def test_exc_message_preserves_chained_cause() -> None:
    """Helper does not interfere with ``raise ... from e`` chaining."""
    try:
        try:
            raise RuntimeError()
        except RuntimeError as e:
            raise click.ClickException(_exc_message(e)) from e
    except click.ClickException as ce:
        assert ce.message == "RuntimeError"
        assert isinstance(ce.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Wrapper-rendering regression tests
#
# The bug class is identical across all in-scope wrappers (ClickException
# and KaguraConnectionError): ``f"{prefix}: {e}"`` strands the prefix when
# ``str(e) == ""``. Each tuple below mirrors a prefix used by an actual
# wrapper call site so the parametrize matrix gives evidence for every
# rendered prefix even though the underlying invariant lives in
# ``_exc_message``.
# ---------------------------------------------------------------------------


def _empty_runtime() -> RuntimeError:
    return RuntimeError()


def _empty_httpx_connect() -> httpx.ConnectError:
    return httpx.ConnectError("")


def _empty_value() -> ValueError:
    return ValueError()


# Each row pins a prefix actually used by a wrapper in the codebase, paired
# with a factory whose ``str()`` is empty in a way real upstream code can
# surface (e.g. ``httpx.ConnectError("")``).
_PREFIXED_SITES: list[tuple[str, Callable[[], BaseException], str]] = [
    # ClickException prefixes (cli.py + setup_claude.py + auth/cli.py)
    ("Setup failed", _empty_runtime, "RuntimeError"),
    ("Connection failed", _empty_runtime, "RuntimeError"),
    ("Authentication failed", _empty_runtime, "RuntimeError"),
    ("Failed to read input", _empty_runtime, "RuntimeError"),
    # KaguraConnectionError prefixes (auth/device_flow.py, client.py,
    # files_client.py, resource_client.py)
    ("HTTP 500", _empty_runtime, "RuntimeError"),
    ("Could not reach https://example.com", _empty_httpx_connect, "ConnectError"),
    ("Lost connection while waiting for approval", _empty_httpx_connect, "ConnectError"),
    ("Object store PUT failed", _empty_httpx_connect, "ConnectError"),
    ("Object store PUT timed out", _empty_httpx_connect, "ConnectError"),
    ("Invalid response format", _empty_value, "ValueError"),
]


@pytest.mark.parametrize("prefix, exc_factory, expected_class", _PREFIXED_SITES)
def test_prefixed_wrapper_never_strands_prefix(
    prefix: str,
    exc_factory: Callable[[], BaseException],
    expected_class: str,
) -> None:
    """``f'{prefix}: {_exc_message(e)}'`` always carries non-empty content past the colon."""
    e = exc_factory()
    rendered = f"{prefix}: {_exc_message(e)}"
    assert rendered == f"{prefix}: {expected_class}"
    assert not rendered.endswith(": ")
    assert not rendered.endswith(":\n")
    assert rendered.split(": ", 1)[1] != ""


def test_bare_wrapper_never_renders_empty_message() -> None:
    """``ClickException(_exc_message(e))`` — the prefix-less wrapper — also never empties out."""
    e = RuntimeError()
    msg = _exc_message(e)
    assert msg == "RuntimeError"
    assert msg != ""
    rendered = click.ClickException(msg).message
    assert rendered == "RuntimeError"
