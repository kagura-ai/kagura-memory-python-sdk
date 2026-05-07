"""Tests for ingest._safety: IP denylist and system path denylist."""

from pathlib import Path

import pytest

from kagura_memory.ingest._safety import is_blocked_ip, is_blocked_system_path


class TestIsBlockedIp:
    """The IP denylist is the SSRF guard's last line of defense."""

    @pytest.mark.parametrize(
        "ip",
        [
            # IPv4 loopback
            "127.0.0.1",
            "127.10.20.30",
            # RFC 1918
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            # Link-local (cloud IMDS)
            "169.254.169.254",
            "169.254.0.1",
            # CGNAT / shared
            "100.64.0.1",
            # Multicast / reserved
            "224.0.0.1",
            "239.255.255.255",
            "255.255.255.255",
            # IPv6 loopback / unique-local / link-local
            "::1",
            "fc00::1",
            "fd00::abcd",
            "fe80::1",
            # IPv6 multicast
            "ff02::1",
            # IPv4-mapped IPv6
            "::ffff:10.0.0.1",
            "::ffff:127.0.0.1",
        ],
    )
    def test_blocked_ips(self, ip: str) -> None:
        assert is_blocked_ip(ip), f"expected {ip} to be blocked"

    @pytest.mark.parametrize(
        "ip",
        [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",  # example.com
            "2606:4700:4700::1111",  # cloudflare DNS v6
            "2a00:1450:4001:81b::200e",  # public IPv6
        ],
    )
    def test_public_ips_allowed(self, ip: str) -> None:
        assert not is_blocked_ip(ip), f"expected {ip} to be allowed"

    @pytest.mark.parametrize("garbage", ["", "not-an-ip", "999.999.999.999", "::xyz"])
    def test_malformed_treated_as_blocked(self, garbage: str) -> None:
        # Conservative: anything we can't parse is rejected.
        assert is_blocked_ip(garbage), f"expected malformed {garbage!r} to be blocked"


class TestIsBlockedSystemPath:
    """Local file ingestion default-deny on sensitive system directories."""

    @pytest.mark.parametrize(
        "path",
        [
            "/etc/passwd",
            "/etc/shadow",
            "/proc/cpuinfo",
            "/sys/class/net",
            "/dev/random",
            "/root/.bashrc",
            "/boot/grub.cfg",
            "/var/log/auth.log",
            "/.git/config",
        ],
    )
    def test_blocked_paths(self, path: str) -> None:
        assert is_blocked_system_path(Path(path))

    def test_user_ssh_blocked(self) -> None:
        # The home-relative SSH dir is dynamically built from Path.home().
        ssh_path = Path.home() / ".ssh" / "id_rsa"
        assert is_blocked_system_path(ssh_path)

    @pytest.mark.parametrize(
        "path",
        [
            "/tmp/document.pdf",
            "/home/user/Documents/report.pdf",
            "/var/folders/some/cache.pdf",
            "/etc-not-really/file.pdf",  # prefix substring shouldn't match
        ],
    )
    def test_safe_paths_allowed(self, path: str) -> None:
        assert not is_blocked_system_path(Path(path))

    def test_relative_path_not_blocked(self) -> None:
        # Relative paths are deferred — caller must resolve them first.
        # Treating relative as "blocked" would cause false positives during
        # tests run from arbitrary cwd.
        assert not is_blocked_system_path(Path("relative/file.pdf"))
