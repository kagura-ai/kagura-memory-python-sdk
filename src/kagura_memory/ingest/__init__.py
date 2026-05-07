"""File ingestion module for Kagura Memory SDK (Issue #80).

Convert URLs and local files (PDF, image — Phase 1) into structured Memory
Cloud memories. The high-level entry point is :class:`FileIngestor`; the
``kagura ingest`` CLI subcommand is a thin wrapper around it.

The public surface is intentionally small:

* :class:`FileIngestor` — orchestrator
* :class:`Fetcher` — SSRF-hardened URL/file fetch
* :class:`Provider` — text/vision LLM Protocol (implementations: Claude, Gemini, Ollama)
* :class:`Extractor` — structural-extractor Protocol (implementations: PDF)

Result types (:class:`IngestResult`, :class:`CostBreakdown`,
:class:`IngestErrorRecord`) live in :mod:`kagura_memory.models` and are
re-exported at the package root for convenience.

Heavy parsing dependencies (``pymupdf``, ``pillow``) are lazy-imported
inside the concrete extractors. Importing this module never triggers them.
"""

from .extractors.base import Extractor
from .fetcher import Fetcher, FetchResult
from .ingestor import FileIngestor
from .providers.base import Provider

__all__ = [
    "FileIngestor",
    "Fetcher",
    "FetchResult",
    "Extractor",
    "Provider",
]
