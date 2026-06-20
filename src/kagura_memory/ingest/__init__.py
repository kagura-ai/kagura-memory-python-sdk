"""File ingestion module for Kagura Memory SDK (Issue #80).

Convert URLs and local files into structured Memory Cloud memories. The
high-level entry point is :class:`FileIngestor`; the ``kagura ingest`` CLI
subcommand is a thin wrapper around it.

Supported inputs: PDF, plain text / Markdown, HTML, DOCX, XLSX, PPTX, EPUB,
images (vision), audio/video transcription, and YouTube transcripts; a URL may
optionally be rendered with a headless browser for JS-heavy pages. Each
parser's heavy dependency is opt-in via a ``kagura-memory[ingest-*]`` extra.

The public surface is intentionally small:

* :class:`FileIngestor` — orchestrator
* :class:`Fetcher` — SSRF-hardened URL/file fetch
* :class:`Provider` — text/vision LLM Protocol (implementations: Claude, Gemini, Ollama)
* :class:`Extractor` — structural-extractor Protocol (implementations: PDF,
  text/Markdown, HTML, DOCX, XLSX, PPTX, EPUB)

Result types (:class:`IngestResult`, :class:`CostBreakdown`,
:class:`IngestErrorRecord`) live in :mod:`kagura_memory.models` and are
re-exported at the package root for convenience.

Heavy parsing dependencies (``pymupdf``, ``pillow``, ``beautifulsoup4``,
``python-docx``, ``openpyxl``, ``python-pptx``) are lazy-imported inside the
concrete extractors. Importing this module never triggers them.
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
