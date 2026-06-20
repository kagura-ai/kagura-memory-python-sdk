"""SSRF and path-safety guards for the file ingestion fetcher.

The IP denylist is conservative: it blocks anything that could refer to the
local host, the local network, or cloud-provider metadata endpoints. The
list is built from RFC 1918, RFC 4193, RFC 6890, and the documented cloud
metadata IP ranges as of 2026-Q2.

DNS rebinding: pre-flight resolution + this denylist rejects an
externally-supplied hostname that points at, e.g., ``169.254.169.254``. The
timing-correlated rebinding window (a hostname that flips between our resolve
and httpx's connect) is closed by the fetcher pinning the connection to the
exact IP validated here — see :meth:`Fetcher._stream_request` (#188).
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path

_BLOCKED_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),  # "this network", spec says don't route
    ipaddress.ip_network("127.0.0.0/8"),  # IPv4 loopback
    ipaddress.ip_network("10.0.0.0/8"),  # RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),  # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918
    ipaddress.ip_network("169.254.0.0/16"),  # link-local incl. AWS/GCP/Azure IMDS
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT / shared address space (RFC 6598)
    ipaddress.ip_network("224.0.0.0/4"),  # IPv4 multicast
    ipaddress.ip_network("240.0.0.0/4"),  # reserved (includes 255.255.255.255)
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped IPv6
    ipaddress.ip_network("ff00::/8"),  # IPv6 multicast
)

_DEFAULT_BLOCKED_PATH_PREFIXES: tuple[str, ...] = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/root",
    "/boot",
    "/var/log",
    "/var/lib",
    "/.git",
)


def is_blocked_ip(ip_str: str) -> bool:
    """True iff ``ip_str`` falls in any blocked network.

    Args:
        ip_str: IP address as a string (IPv4 or IPv6).

    Returns:
        ``True`` if the address is in a blocked range, ``False`` otherwise.
        Malformed IPs are conservatively treated as blocked.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return any(ip in net for net in _BLOCKED_NETWORKS)


def _blocked_prefixes() -> tuple[Path, ...]:
    """Sensitive directory prefixes to default-deny for the running platform.

    ``~/.ssh`` is always included (resolved against the current ``$HOME``). On
    Windows the system directory (``%SystemRoot%``, typically ``C:\\Windows``)
    and ``%ProgramData%`` are blocked; on POSIX the directories in
    :data:`_DEFAULT_BLOCKED_PATH_PREFIXES` are used. The POSIX list is left out
    on Windows because a drive-rooted path can never live under ``/etc`` et al.
    """
    prefixes: list[Path] = [Path.home() / ".ssh"]
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
        prefixes.append(Path(system_root))
        program_data = os.environ.get("ProgramData")
        if program_data:
            prefixes.append(Path(program_data))
    else:
        prefixes.extend(Path(p) for p in _DEFAULT_BLOCKED_PATH_PREFIXES)
    return tuple(prefixes)


def is_blocked_system_path(path: Path) -> bool:
    """True iff the absolute path is inside a sensitive system directory.

    Args:
        path: Filesystem path to test. Must already be resolved to an
            absolute path; this function does NOT call ``.resolve()``
            because callers may want to log the original (pre-resolve)
            path on rejection.

    Returns:
        ``True`` if the absolute path is at or under a default-blocked system
        directory, ``False`` otherwise.

    Matching uses :meth:`pathlib.PurePath.is_relative_to`, so a prefix only
    matches on a path-segment boundary — ``/var/log`` blocks ``/var/log/x`` but
    not ``/varlog`` — and comparison is case-insensitive on Windows.
    """
    if not path.is_absolute():
        return False
    return any(path.is_relative_to(prefix) for prefix in _blocked_prefixes())
