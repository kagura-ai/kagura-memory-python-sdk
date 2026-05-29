"""Tests for the extractor registry (extractors/__init__.py).

These run without any optional parser dep installed: ``supported_mimes``
and the registry-vs-``supports`` invariant only touch the static table and
each extractor module's class attribute — no heavy import is triggered for
the formats whose deps are present, and missing deps surface only when
``extract()`` is called (not at class definition time).
"""

from __future__ import annotations

import pytest

from kagura_memory.ingest.extractors import (
    _REGISTRY,
    DOCX_MIME,
    EPUB_MIME,
    PPTX_MIME,
    XLSX_MIME,
    get_extractor,
    supported_mimes,
)


def test_supported_mimes_lists_all_registered_types() -> None:
    assert supported_mimes() == {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/html",
        DOCX_MIME,
        XLSX_MIME,
        PPTX_MIME,
        EPUB_MIME,
    }


def test_get_extractor_unknown_mime_raises_value_error_listing_supported() -> None:
    with pytest.raises(ValueError, match="no extractor registered for MIME") as exc:
        get_extractor("application/x-nonsense")
    # The error enumerates the registered types (acceptance criterion).
    assert "application/pdf" in str(exc.value)
    assert "text/markdown" in str(exc.value)


def test_get_extractor_is_case_insensitive_on_mime() -> None:
    # text/* needs no optional dep, so this import always succeeds.
    extractor = get_extractor("TEXT/Markdown")
    assert "text/markdown" in extractor.supports


@pytest.mark.parametrize("mime", sorted(_REGISTRY))
def test_registry_class_supports_includes_its_mime(mime: str) -> None:
    """Each registered class's ``supports`` set must contain its registry MIME.

    Guards against drift between the static ``_REGISTRY`` table and the
    per-extractor ``supports`` declaration. Skips a format whose optional
    parser dep is not importable in this environment.
    """
    module_name, class_name = _REGISTRY[mime]
    pytest.importorskip(
        f"kagura_memory.ingest.extractors.{module_name}",
        reason=f"optional dep for {module_name} extractor not installed",
    )
    from importlib import import_module

    module = import_module(f"kagura_memory.ingest.extractors.{module_name}")
    extractor_cls = getattr(module, class_name)
    assert mime in extractor_cls.supports
