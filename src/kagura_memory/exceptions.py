"""Custom exceptions for Kagura Memory SDK."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import RollbackSummary


def _exc_message(e: BaseException) -> str:
    """Return ``str(e)`` when non-empty, otherwise the class name.

    Defensive fallback so an unmessaged exception (e.g. ``raise
    RuntimeError()``) still produces a non-empty diagnostic when
    interpolated into a user-facing error string. Used by the SDK
    wherever a ``ClickException`` or ``KaguraConnectionError`` wrapper
    interpolates ``{e}`` / ``str(e)`` into the message it raises.
    Originating bug: #127.
    """
    return str(e) or e.__class__.__name__


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


class KaguraResponseError(KaguraError):
    """A successful server response did not match the SDK's model of it (#250).

    Usually the server is newer than this SDK and returns a value or a
    shape the SDK does not know yet; upgrading ``kagura-memory`` is the
    likely fix. ``operation`` names the call whose response failed to
    parse — the MCP tool name, or ``<Client>.<method>`` on the REST
    clients — and prefixes the message.

    The message describes the failing fields or the envelope shape, not
    payload values. When a model rejected the payload, the
    ``pydantic.ValidationError`` is chained as ``__cause__``, and *its*
    text does include input values, so a logged traceback can show
    payload contents. One deliberate exception:
    ``WorkspaceClient.mint_member_key`` puts the one-time plaintext key in
    its message, because the key was created and cannot be shown again.
    """

    def __init__(self, message: str, operation: str | None = None):
        super().__init__(message)
        self.operation = operation


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
    """A quota or plan cap was reached (#256).

    Raised for the server's quota refusals: a daily or rolling quota
    (``memories_per_day``, analysis runs, the resource-token events-per-hour
    ceiling, the daily MCP call cap on the ``KaguraClient`` tool methods,
    the daily REST quota on the REST clients) and a count cap (contexts,
    members, resource tokens, connectors, agents, storage). The REST
    clients other than ``SecretClient`` also raise it for any other HTTP
    429, such as the per-minute rate limit; ``SecretClient`` keeps its
    generic :class:`KaguraConnectionError` for those. ``KaguraClient``'s
    HTTP-level 429 — the per-minute limit, or the daily MCP quota when the
    server's rate-limit middleware refuses the request itself — raises
    :class:`KaguraRateLimitError` instead.

    memory-cloud v0.75.0+ marks these refusals with ``gate="quota"``; the
    detail attributes are read from the MCP envelope or the REST
    ``details``, and are ``None`` when the server did not send them (older
    servers send only ``quota_type``, ``limit``, ``used_today`` and
    ``resets_at``). A refusal with no ``gate``, no known ``quota_type``
    and no retry window, such as the 1 MB memory-size guard, is a request
    limit no plan lifts and raises a plain :class:`KaguraError` instead.

    Attributes:
        retry_after: Seconds to wait before retrying — the server's
            ``Retry-After`` when it sent one, else derived from
            ``resets_at`` when the error is raised. ``None`` for a count
            cap, where waiting does not help: free capacity or upgrade
            instead.
        quota_type: Which quota, e.g. ``"memories_per_day"``,
            ``"contexts"``, ``"resource_tokens"``, ``"members"``.
        limit: The cap that was hit.
        current: The count already used (server v0.75.0+).
        used_today: The day's count on the daily quotas — the legacy name
            of ``current``, which older servers send instead.
        resets_at: When a time-windowed quota resets (timezone-aware).
        gate: ``"quota"`` on server v0.75.0+, else ``None``.
        feature: The feature the cap belongs to, when it belongs to one.
        required_plan: Plan key of the lowest tier that raises the cap, or
            ``None`` when no tier does.
        required_plan_display: Display label of ``required_plan``, e.g.
            ``"L"``.
        current_plan: The workspace's plan key.
        details: Every field the server sent with the refusal — the MCP
            envelope without ``status`` / ``error`` / ``message``, or the
            REST ``details`` — including those with no attribute here, e.g.
            ``period`` / ``cap_usd`` / ``current_usd`` on the embedding
            spend cap. Empty when the SDK raised the error itself.
    """

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        *,
        quota_type: str | None = None,
        limit: int | None = None,
        current: int | None = None,
        used_today: int | None = None,
        resets_at: datetime | None = None,
        gate: str | None = None,
        feature: str | None = None,
        required_plan: str | None = None,
        required_plan_display: str | None = None,
        current_plan: str | None = None,
        details: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        if resets_at is not None and resets_at.tzinfo is None:
            resets_at = resets_at.replace(tzinfo=UTC)
        if retry_after is None and resets_at is not None:
            seconds = (resets_at - datetime.now(UTC)).total_seconds()
            retry_after = max(0, math.ceil(seconds))
        self.retry_after = retry_after
        self.quota_type = quota_type
        self.limit = limit
        self.current = current
        self.used_today = used_today
        self.resets_at = resets_at
        self.gate = gate
        self.feature = feature
        self.required_plan = required_plan
        self.required_plan_display = required_plan_display
        self.current_plan = current_plan
        self.details: dict[str, Any] = dict(details or {})


class KaguraFeatureNotAvailableError(KaguraError):
    """The workspace cannot use a feature (#256).

    Raised for MCP ``plan_required`` / ``feature_not_available`` and REST
    ``FEAT-001``, and for any refusal whose ``gate`` is ``"plan"``,
    ``"allowlist"`` or ``"deployment"`` (memory-cloud v0.75.0+).
    ``gate`` says why: the plan does not include the feature (an upgrade
    lifts it, see ``required_plan``), a rollout allowlist excludes the
    workspace, or the operator turned the feature off on this deployment —
    the last two have no upgrade path. ``gate`` is ``None`` on older
    servers, which send only ``required_plan`` (MCP) or ``feature`` (REST).

    Attributes:
        feature: Registry key of the feature, e.g. ``"resources"``.
        required_plan: Plan key of the lowest tier with the feature, or
            ``None`` when no tier has it (or the gate is not ``"plan"``).
        required_plan_display: Display label of ``required_plan``, e.g.
            ``"XL"``.
        current_plan: The workspace's plan key.
        gate: ``"plan"``, ``"allowlist"`` or ``"deployment"``.
        details: Every field the server sent with the refusal, as on
            :class:`KaguraQuotaError`.
    """

    def __init__(
        self,
        message: str,
        *,
        feature: str | None = None,
        required_plan: str | None = None,
        required_plan_display: str | None = None,
        current_plan: str | None = None,
        gate: str | None = None,
        details: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.feature = feature
        self.required_plan = required_plan
        self.required_plan_display = required_plan_display
        self.current_plan = current_plan
        self.gate = gate
        self.details: dict[str, Any] = dict(details or {})


class KaguraPartialRollbackError(KaguraError):
    """``rollback_sleep_run`` reversed only part of the run (#256).

    Some undo steps failed; the server marks the report ``failed``. The
    steps that succeeded stay reversed, and ``summary`` counts them per
    category, with the failures in ``summary.errors`` — read it to decide
    whether to retry.

    Attributes:
        report_id: The Sleep report that was rolled back.
        summary: What was reversed, kept and failed. ``None`` when the
            server's ``rollback_summary`` was missing or did not parse; the
            parse error is then chained as ``__cause__``.
    """

    def __init__(
        self,
        message: str,
        *,
        report_id: str | None = None,
        summary: RollbackSummary | None = None,
    ):
        super().__init__(message)
        self.report_id = report_id
        self.summary = summary


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


class KaguraSecretError(KaguraError):
    """Base error for the zero-knowledge secret client (Issue #216)."""


class KaguraCryptoError(KaguraSecretError):
    """age encryption/decryption or armor (de)framing failed.

    Raised by :mod:`kagura_memory.secrets.crypto` for malformed recipients,
    decrypt failures (wrong identity / corrupt ciphertext), non-armored input,
    or a ciphertext that exceeds the server's size cap. The underlying
    ``pyrage`` error (if any) is preserved via ``__cause__``.
    """


class KaguraKeyCustodyError(KaguraSecretError):
    """The age private key could not be stored or retrieved securely.

    Raised by :mod:`kagura_memory.secrets.keymanager` when no acceptable
    custody backend is available (fail-closed) or the keychain rejects a
    read/write. Repo-dotfile private keys are disallowed by design.
    """
