"""Tests for ingest._safety — SSRF IP denylist + path-traversal guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from kagura_memory.ingest._safety import is_blocked_ip, is_blocked_system_path


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


def test_is_blocked_system_path_blocks_etc() -> None:
    assert is_blocked_system_path(Path("/etc/passwd")) is True
    assert is_blocked_system_path(Path("/etc")) is True


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


def test_is_blocked_system_path_distinguishes_prefix_match_from_exact() -> None:
    """'/varlog' MUST NOT match the '/var/log' prefix — requires separator."""
    assert is_blocked_system_path(Path("/varlog")) is False
    assert is_blocked_system_path(Path("/var/log")) is True
    assert is_blocked_system_path(Path("/var/log/anything.log")) is True
