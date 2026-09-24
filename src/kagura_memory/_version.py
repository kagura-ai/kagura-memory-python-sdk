"""The SDK's one version parser: memory-cloud server versions and PyPI versions.

``kagura doctor``, :meth:`KaguraClient.check_server_version` and
``device_flow.invite_support`` compare a version string with a minimum, and
the LiteLLM check in ``kagura doctor`` compares one with blocked releases.
They all read the string here, so one string gets one verdict.

A version is ``v?MAJOR.MINOR.PATCH`` at the start of the string, in ASCII
digits, with letters read in any case as PEP 440 does (``V1.2.3``,
``.POST1``). What follows the triple says whether it is a release or a
pre-release:

- **Release:** nothing, ``+build`` (SemVer build metadata, a PEP 440 local
  version), more ``.N`` components (``1.2.3.4``) or ``.postN``.
- **Pre-release:** anything else, e.g. ``-rc1`` (SemVer), ``rc1`` / ``a1`` /
  ``b2`` / ``.dev1`` (PEP 440), or stray text such as ``.rc1`` or ``_x``.
  That includes PEP 440's other post-release spellings (``post1``,
  ``-post1``, ``-1``, ``r1``, ``.rev1``): only ``.postN`` is read as one.

A pre-release of a triple comes before that triple (SemVer §11, PEP 440) and
after every lower one. Anything else is unparseable: a non-``str``, fewer
than three components (``0.76``), or text such as ``main-abc123``.
"""

from __future__ import annotations

import re

# At most 32 significant digits per component, so a longer one is unparseable:
# past its str-digit limit (4300 by default) int() raises ValueError, and a
# version a server sends must never make a check raise. Leading zeros are
# skipped, not counted, as PEP 440 reads them ("1.82.0007" is 1.82.7), and a
# run of them is a lone 0 so a mismatch backtracks in linear time. ``(?!\d)``
# keeps a longer patch from being cut to its first 32 digits.
_COMPONENT = r"0*([1-9]\d{0,31}|0)"
_VERSION_RE = re.compile(
    rf"v?{_COMPONENT}\.{_COMPONENT}\.{_COMPONENT}(?!\d)", re.ASCII | re.IGNORECASE
)
_RELEASE_TAIL_RE = re.compile(
    r"(?:\.\d+)*(?:\.post\d*)?(?:\+.*)?", re.ASCII | re.IGNORECASE | re.DOTALL
)


def _parse(value: object) -> tuple[tuple[int, int, int], bool] | None:
    """``((major, minor, patch), is_prerelease)``, or ``None`` when unparseable."""
    if not isinstance(value, str):
        return None
    match = _VERSION_RE.match(value)
    if match is None:
        return None
    triple = (int(match[1]), int(match[2]), int(match[3]))
    return triple, _RELEASE_TAIL_RE.fullmatch(value, match.end()) is None


def parse_version(value: object) -> tuple[int, int, int] | None:
    """The ``(major, minor, patch)`` a version string starts with.

    A pre-release returns its triple too (``"0.76.0-rc1"`` → ``(0, 76, 0)``);
    use :func:`meets_minimum` to compare against a minimum.

    Args:
        value: The version, e.g. ``ServerInfo.version``. Any type is accepted.

    Returns:
        The triple, or ``None`` when ``value`` is not a ``str`` or does not
        start with ``v?MAJOR.MINOR.PATCH``.
    """
    parsed = _parse(value)
    return None if parsed is None else parsed[0]


def meets_minimum(value: object, minimum: tuple[int, int, int]) -> bool | None:
    """Whether a version is at least ``minimum``.

    A pre-release of exactly ``minimum`` is below it (``"0.17.1-rc1"`` does
    not meet ``(0, 17, 1)``); a pre-release of a higher triple is above it
    (``"0.17.2-rc1"`` does).

    Args:
        value: The version, e.g. ``ServerInfo.version``. Any type is accepted.
        minimum: The lowest acceptable release.

    Returns:
        ``True`` or ``False``, or ``None`` when ``value`` is unparseable (see
        :func:`parse_version`), so the caller picks its own fallback.
    """
    parsed = _parse(value)
    if parsed is None:
        return None
    triple, is_prerelease = parsed
    if triple == minimum:
        return not is_prerelease
    return triple > minimum
