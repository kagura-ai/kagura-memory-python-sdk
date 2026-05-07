"""Custom exceptions for Kagura Memory SDK."""


class KaguraError(Exception):
    """Base exception for Kagura SDK."""


class KaguraAuthError(KaguraError):
    """Authentication failed."""


class KaguraConnectionError(KaguraError):
    """Connection to Kagura server failed."""


class KaguraNotFoundError(KaguraError):
    """Requested resource not found (HTTP 404)."""


class KaguraRateLimitError(KaguraError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class KaguraLLMError(KaguraError):
    """LLM call failed."""


class KaguraContextError(KaguraError):
    """Context not found or invalid."""


class KaguraQuotaError(KaguraError):
    """Resource token quota exceeded (events per hour)."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class KaguraFetchError(KaguraError):
    """URL or file fetch failed (SSRF guard, byte cap, redirect loop, etc.).

    Raised by the file-ingestion fetcher when a source URL or path cannot be
    safely retrieved. The original URL/path is exposed via the ``url`` attribute
    so callers can present it without re-parsing the message.
    """

    def __init__(self, message: str, url: str | None = None):
        super().__init__(message)
        self.url = url


class KaguraIngestError(KaguraError):
    """File ingestion orchestration failed for a non-fetch reason.

    Used for extractor failures, provider failures, and chunker failures. Per-
    section partial failures are reported via ``IngestResult.errors`` (best
    effort) and do NOT raise this exception — only fatal orchestration failures
    do (e.g. extractor cannot decode the file, no Provider configured).
    """
