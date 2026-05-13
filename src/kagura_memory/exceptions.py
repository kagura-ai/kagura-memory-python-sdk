"""Custom exceptions for Kagura Memory SDK."""

from datetime import datetime


class KaguraError(Exception):
    """Base exception for Kagura SDK."""


class KaguraAuthError(KaguraError):
    """Authentication failed."""


class KaguraAuthExpiredError(KaguraAuthError):
    """OAuth refresh token expired or invalid.

    Raised when an attempted refresh returns ``invalid_grant`` (or the
    server otherwise indicates that the stored refresh token can no
    longer be used). The caller must re-authenticate via
    ``kagura auth login``.
    """

    def __init__(self, message: str, expires_at: datetime | None = None):
        super().__init__(message)
        self.expires_at = expires_at


class KaguraAuthDeniedError(KaguraAuthError):
    """User denied authorization at the device-flow consent screen."""


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


class KaguraIntegrityError(KaguraError):
    """Object store rejected an upload with HTTP 400.

    Raised for any ``HTTP 400`` response from the object store on a
    presigned PUT — most commonly R2 ``BadDigest`` (the body's sha256
    did not match the value bound into the presigned PUT URL via the
    ``x-amz-checksum-sha256`` header / ``ChecksumSHA256`` parameter),
    but also covers other 400 causes such as a malformed presigned
    URL or a ``Content-Length`` mismatch. The exception message
    documents the most likely cause; callers should not assume a
    specific S3-XML error code without inspecting the underlying
    ``__cause__`` response body.
    """
