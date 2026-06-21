"""Tests for ingest._safety — SSRF IP denylist + path-traversal guard."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from kagura_memory.ingest import _safety
from kagura_memory.ingest._safety import (
    _blocked_prefixes,
    is_blocked_ip,
    is_blocked_system_path,
)

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX system paths (/etc, /proc, ...) are not absolute on Windows",
)
_WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows-specific system-path blocking"
)


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.5.5",
        "192.168.1.1",
        "169.254.169.254",  # AWS / GCP / Azure IMDS
        "100.64.0.5",  # CGNAT
        "::1",  # IPv6 loopback
        "fe80::1",  # IPv6 link-local
        "fc00::1",  # IPv6 ULA
        "::ffff:127.0.0.1",  # IPv4-mapped IPv6
    ],
)
def test_is_blocked_ip_blocks_internal_ranges(ip: str) -> None:
    assert is_blocked_ip(ip) is True


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_is_blocked_ip_allows_public_ranges(ip: str) -> None:
    assert is_blocked_ip(ip) is False


def test_is_blocked_ip_treats_malformed_as_blocked() -> None:
    """Conservative fail-secure: garbage strings are treated as blocked."""
    assert is_blocked_ip("not-an-ip") is True
    assert is_blocked_ip("") is True
    assert is_blocked_ip("999.999.999.999") is True


@_POSIX_ONLY
def test_is_blocked_system_path_blocks_etc() -> None:
    assert is_blocked_system_path(Path("/etc/passwd")) is True
    assert is_blocked_system_path(Path("/etc")) is True


@_POSIX_ONLY
def test_is_blocked_system_path_blocks_proc() -> None:
    assert is_blocked_system_path(Path("/proc/1/maps")) is True


def test_is_blocked_system_path_blocks_ssh_dir() -> None:
    """~/.ssh is dynamically resolved against $HOME and blocked."""
    ssh_path = Path.home() / ".ssh" / "id_rsa"
    assert is_blocked_system_path(ssh_path) is True


def test_is_blocked_system_path_allows_user_path() -> None:
    assert is_blocked_system_path(Path("/tmp/foo.pdf")) is False
    assert is_blocked_system_path(Path("/home/user/docs/report.pdf")) is False


def test_is_blocked_system_path_returns_false_for_relative_path() -> None:
    """Non-absolute paths return False — caller resolves before checking."""
    assert is_blocked_system_path(Path("relative/path.txt")) is False


@_POSIX_ONLY
def test_is_blocked_system_path_distinguishes_prefix_match_from_exact() -> None:
    """'/varlog' MUST NOT match the '/var/log' prefix — requires separator."""
    assert is_blocked_system_path(Path("/varlog")) is False
    assert is_blocked_system_path(Path("/var/log")) is True
    assert is_blocked_system_path(Path("/var/log/anything.log")) is True


def test_blocked_prefixes_includes_ssh_dir() -> None:
    """~/.ssh is always blocked, regardless of platform."""
    assert (Path.home() / ".ssh") in _blocked_prefixes()


def test_blocked_prefixes_windows_branch(monkeypatch) -> None:
    """Cover the Windows prefix branch on any host (CI runs Linux).

    Inject a fake ``os`` into the module rather than patching the real
    ``os.name`` — the latter corrupts pytest's own platform checks.
    """
    fake_os = types.SimpleNamespace(
        name="nt",
        environ={"SystemRoot": r"C:\Windows", "ProgramData": r"C:\ProgramData"},
    )
    monkeypatch.setattr(_safety, "os", fake_os)
    joined = " ".join(str(p) for p in _blocked_prefixes())
    assert "Windows" in joined
    assert "ProgramData" in joined


def test_blocked_prefixes_posix_branch(monkeypatch) -> None:
    """Cover the POSIX prefix branch on any host (dev runs Windows)."""
    monkeypatch.setattr(_safety, "os", types.SimpleNamespace(name="posix", environ={}))
    joined = " ".join(str(p) for p in _blocked_prefixes())
    assert "etc" in joined and "proc" in joined  # POSIX system dirs present


@_WINDOWS_ONLY
def test_is_blocked_system_path_blocks_windows_system_root() -> None:
    """The Windows directory (and its children) are blocked, case-insensitively."""
    import os

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    assert is_blocked_system_path(system_root) is True
    assert is_blocked_system_path(system_root / "System32" / "config" / "SAM") is True
    # Case-insensitive match (NTFS is case-insensitive).
    assert is_blocked_system_path(Path(str(system_root).lower())) is True


@_WINDOWS_ONLY
def test_is_blocked_system_path_allows_windows_user_path() -> None:
    """A normal user document path is not blocked on Windows."""
    assert is_blocked_system_path(Path(r"C:\Users\someone\docs\report.pdf")) is False
