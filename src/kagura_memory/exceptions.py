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
