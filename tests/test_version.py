"""Tests for :mod:`kagura_memory._version`, the SDK's one version parser (#276).

``doctor``, ``KaguraClient.check_server_version`` and ``invite_support`` each
parsed version strings their own way, so one string could get two verdicts in
a single ``kagura doctor`` run. The table below is the one place the parsing
rules live; the per-caller tests only check how each caller reads the answer.
"""

from __future__ import annotations

import pytest

from kagura_memory._version import meets_minimum, parse_version

# (value, parse_version(value), is_release). ``is_release`` is None for an
# unparseable value; otherwise it is what meets_minimum(value, own triple)
# returns: a release meets its own triple, a pre-release comes before it.
_TABLE: list[tuple[object, tuple[int, int, int] | None, bool | None]] = [
    ("0.76.0", (0, 76, 0), True),
    ("v0.76.0", (0, 76, 0), True),
    ("V0.76.0", (0, 76, 0), True),  # PEP 440 reads letters case-insensitively
    ("0.75.12", (0, 75, 12), True),
    ("20260924.1.0", (20260924, 1, 0), True),
    # Leading zeros are skipped, as PEP 440 reads them, and do not count
    # towards the 32-digit cap below.
    ("0.076.00", (0, 76, 0), True),
    ("0" * 40 + "1.2.3", (1, 2, 3), True),
    ("1.2." + "0" * 40 + "3", (1, 2, 3), True),
    ("0" * 5000 + ".0.0", (0, 0, 0), True),
    ("1" * 32 + ".0.0", (int("1" * 32), 0, 0), True),
    # Build metadata / PEP 440 local version: a release.
    ("1.0.0+build", (1, 0, 0), True),
    ("0.76.0+build.7", (0, 76, 0), True),
    # Extra numeric components and post-releases: a release of the triple.
    ("1.2.3.4", (1, 2, 3), True),
    ("1.2.3.4.5", (1, 2, 3), True),
    ("1.82.7.post1", (1, 82, 7), True),
    ("1.82.7.post", (1, 82, 7), True),
    ("1.82.7.POST1", (1, 82, 7), True),
    ("1.82.7.post1+local", (1, 82, 7), True),
    # SemVer pre-release: anything after "-".
    ("0.76.0-rc1", (0, 76, 0), False),
    ("0.76.1-rc1", (0, 76, 1), False),
    ("0.16.9-beta", (0, 16, 9), False),
    ("0.76.0-1", (0, 76, 0), False),
    # PEP 440 pre-release: a letter right after the patch, or .devN.
    ("0.76.0rc1", (0, 76, 0), False),
    ("0.76.0a1", (0, 76, 0), False),
    ("0.76.0b2", (0, 76, 0), False),
    ("0.76.0.dev1", (0, 76, 0), False),
    # Any other text after the triple also counts as a pre-release.
    ("0.76.0.rc1", (0, 76, 0), False),
    ("0.76.0_x", (0, 76, 0), False),
    ("0.76.0 ", (0, 76, 0), False),
    ("1.2.3.4rc1", (1, 2, 3), False),
    ("1.82.7.post1.dev1", (1, 82, 7), False),
    # Unparseable.
    ("0.76", None, None),
    ("main-abc123", None, None),
    ("unknown", None, None),
    ("", None, None),
    (" 0.76.0", None, None),  # the triple must start the string
    ("vv0.76.0", None, None),
    ("0.76.x", None, None),
    ("０.76.0", None, None),  # FULLWIDTH DIGIT ZERO: ASCII digits only
    ("0.76.٣", None, None),  # ARABIC-INDIC DIGIT THREE
    ("1" * 33 + ".0.0", None, None),  # 33 significant digits
    ("0.76." + "0" * 40 + "1" * 33, None, None),
    ("1" * 5000 + ".0.0", None, None),  # past int()'s str-digit limit: None, not ValueError
    ("0.76." + "1" * 5000, None, None),
    (None, None, None),
    (76, None, None),
    (b"0.76.0", None, None),
    ((0, 76, 0), None, None),
]


@pytest.mark.parametrize(("value", "triple", "is_release"), _TABLE)
def test_parse_version(value, triple, is_release):
    assert parse_version(value) == triple


@pytest.mark.parametrize(("value", "triple", "is_release"), _TABLE)
def test_meets_minimum_at_its_own_triple(value, triple, is_release):
    # With no triple to compare against, use one the value cannot reach.
    assert meets_minimum(value, triple or (0, 0, 0)) is is_release


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0.17.0-rc1", False),
        ("0.17.1-rc1", False),  # a pre-release of the minimum comes before it
        ("0.17.1.dev1", False),
        ("0.17.0", False),
        ("0.9.99", False),  # compared as numbers, not as text
        ("v0.16.0", False),
        ("0.17.1", True),
        ("v0.17.1", True),
        ("0.17.1+build", True),
        ("0.17.1.post1", True),
        # Only the dotted ".postN" is a post-release. PEP 440's other
        # spellings of 0.17.1.post1 fall under "any other text", so they come
        # before the minimum, the conservative side.
        ("0.17.1post1", False),
        ("0.17.1-post1", False),
        ("0.17.1_post1", False),
        ("0.17.1-1", False),
        ("0.17.1r1", False),
        ("0.17.1.rev1", False),
        ("0.17.2-rc1", True),  # a pre-release of a higher triple is above it
        ("0.17.2post1", True),
        ("0.17.10", True),
        ("0.100.0", True),
        ("1.0.0-rc1", True),
        ("unknown", None),
        ("0.17", None),
        ("", None),
        (None, None),
    ],
)
def test_meets_minimum_boundary(value, expected):
    assert meets_minimum(value, (0, 17, 1)) is expected


@pytest.mark.parametrize(
    "value",
    [
        "0.17.1",
        "v0.17.1",
        "0.17.1rc1",
        "0.17.1a1",
        "0.17.1b2",
        "0.17.1.dev1",
        "0.17.1.post1",
        "0.17.1+local",
        "0.17.0",
        "0.17.2rc1",
        "0.17.2.dev0",
        "0.17.10",
        "0.9.99",
        "1.0.0",
        "0.017.001",
        "0.17.0001rc1",
    ],
)
def test_meets_minimum_agrees_with_pep_440(value):
    # For these PEP 440 spellings the answer is PEP 440's own ordering. Its
    # other post-release spellings (0.17.1post1, 0.17.1-1, ...) do not agree:
    # they come before the minimum (see test_meets_minimum_boundary).
    version = pytest.importorskip("packaging.version")
    expected = version.Version(value) >= version.Version("0.17.1")
    assert meets_minimum(value, (0, 17, 1)) is expected
