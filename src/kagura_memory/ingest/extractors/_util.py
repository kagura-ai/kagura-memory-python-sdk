"""Shared helpers for structural extractors.

Keeps the decompression-bomb ceiling and the source-URI title fallback in
one place so every extractor enforces the same safety policy and derives
titles identically.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Decompression-bomb ceiling on *decoded* text, shared by all extractors. A
# single source of truth so tuning this security limit touches one line.
MAX_TOTAL_TEXT_CHARS = 50_000_000  # ~50 MB of decoded text


def filename_title(source_uri: str | None) -> str | None:
    """Best-effort document title from a source URI's basename.

    Strips query/fragment (so ``report.pdf?token=...`` → ``report.pdf``),
    matching the upload-path filename derivation used elsewhere.
    """
    if not source_uri:
        return None
    path = urlsplit(source_uri).path or source_uri
    return path.rsplit("/", 1)[-1] or source_uri
