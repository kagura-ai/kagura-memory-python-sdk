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
    """Base exception for Kagura SDK.

    An error the server reported keeps what it sent. Branch on ``code``
    (and on the subclass), never on the message: memory-cloud words its
    messages for people and may change them.

    Attributes:
        code: The server's error code — the MCP envelope's ``error``
            (``"payload_too_large"``, ``"version_conflict"``,
            ``"capacity_locked"`` …), the REST ``error`` (``"FEAT-001"``,
            ``"CAPACITY-001"`` …) or the OAuth ``error`` — when the SDK read
            one, else ``None``.
        details: The other fields the server sent with the error — the MCP
            envelope without ``status`` / ``error`` / ``message``, the REST
            ``details``, or a JSON-RPC error's ``data``. Empty when the SDK
            raised the error itself or the server sent nothing more. The
            properties below read the MCP error-envelope fields from it.
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.code: str | None = None
        self.details: dict[str, Any] = {}

    @property
    def help(self) -> str | None:
        """The server's next step for the caller (memory-cloud v0.78.0+, #1684)."""
        return _str_detail(self, "help")

    @property
    def cause(self) -> str | None:
        """Why a server failure happened (v0.82.0+, #1742).

        ``"timeout"``, ``"service_unavailable"`` or ``"internal_error"``;
        ``None`` on a refusal, which is the caller's to fix.
        """
        return _str_detail(self, "cause")

    @property
    def correlation_id(self) -> str | None:
        """The id the server logged a failure under — quote it in a report (v0.82.0+)."""
        return _str_detail(self, "correlation_id")

    @property
    def retryable(self) -> bool | None:
        """Whether repeating the call is safe, on a server failure (v0.82.0+).

        ``False`` for a write whose ``outcome`` is ``"unknown"``: check with
        the read ``help`` names before calling it again.
        """
        value = getattr(self, "details", {}).get("retryable")
        return value if isinstance(value, bool) else None

    @property
    def outcome(self) -> str | None:
        """``"unknown"`` when a failed write may or may not have been applied (v0.82.0+)."""
        return _str_detail(self, "outcome")


def _str_detail(err: KaguraError, key: str) -> str | None:
    # getattr: a subclass outside the SDK may skip KaguraError.__init__.
    value = getattr(err, "details", {}).get(key)
    return value if isinstance(value, str) else None


class KaguraAuthError(KaguraError):
    """Authentication failed.

    Also raised for an OAuth token that lacks the scope a call needs
    (HTTP 403 ``insufficient_scope``, memory-cloud v0.79.0+, #1686): ``code``
    is then ``"insufficient_scope"`` and ``details["required_scope"]`` names
    the scope (``"memory:read"`` or ``"memory:write"``) when the server sent
    it. A ``kagura auth login --read-only`` profile gets this on every tool
    that writes; sign in again without ``--read-only``.
    """


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


class KaguraServiceUnavailableError(KaguraConnectionError):
    """The server answered HTTP 503: it could not serve the request right now.

    A :class:`KaguraConnectionError`, so existing handlers still catch it.
    Raised for ``/mcp``'s ``temporarily_unavailable`` (the credential could
    not be verified, memory-cloud v0.82.0+, #1742) and its JSON-RPC
    transport failures (a dependency down or timed out), and for REST
    ``DB-002`` (the database is unavailable). ``details`` keeps what the
    server sent — on ``/mcp`` the JSON-RPC ``data`` with ``cause``,
    ``correlation_id``, ``help``, ``retryable`` and, for a write,
    ``outcome``.

    Attributes:
        retry_after: Seconds to wait, from the ``Retry-After`` header (else
            the JSON-RPC ``data.retry_after_seconds``); ``None`` when the
            server named no wait — on ``/mcp`` that means the request was a
            write whose ``outcome`` is unknown, so check before repeating it.
        reason: Why: REST ``details.reason`` (``pool_exhausted``,
            ``connection_refused``, ``deadlock``, ``database_error``), the
            MCP ``cause`` (``timeout``, ``service_unavailable``), or the error
            code (``temporarily_unavailable``) when nothing finer was sent.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        reason: str | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.reason = reason


class KaguraNotFoundError(KaguraError):
    """Requested resource not found (HTTP 404)."""


class KaguraResponseError(KaguraError):
    """A successful server response did not match the SDK's model of it (#250).

    Usually the server is newer than this SDK and returns a value or a
    shape the SDK does not know yet; upgrading ``kagura-memory`` is the
    likely fix. ``operation`` names the call whose response failed to
    parse — the MCP tool name, or ``<Client>.<method>`` on the REST
    clients and on ``KaguraClient``'s REST-backed methods (#277) — and
    prefixes the message.

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
    and no retry window, such as the REST 1 MB memory-size guard (429
    ``QUOTA-001``), is a request limit no plan lifts and raises a plain
    :class:`KaguraError` instead. On MCP that guard is a plain
    :class:`KaguraError` too: ``code`` ``"payload_too_large"`` with ``limit``
    / ``actual`` in ``details`` on memory-cloud v0.108.0+ (#2003), an untyped
    ``quota_exceeded`` before.

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
        self.details = dict(details or {})


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

    Since memory-cloud v0.100.0 (#1943, #1957) a plan that dropped below what
    the workspace holds is refused this way too: ``feature="members"`` on
    every workspace route for a member without a seat, and
    ``feature="contexts"`` on a content write (remember, update, uploads,
    agent state, edges, measurements, feedback, search settings) into a
    context without a writable slot — reads keep working. ``limit`` /
    ``current`` (seats or slots, and the members or live contexts) are in
    ``details``.

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
        self.details = dict(details or {})


class KaguraCapacityLockedError(KaguraError):
    """The workspace is over its Free plan capacity and locked (memory-cloud v0.99.0+, #1941).

    A workspace back on Free from a subscription that still holds more
    memories or file bytes than Free allows: search and saving (recall,
    reference, explore, remember, uploads, context creation …) are refused
    until it is cleaned up or re-subscribed; listing, deleting, export and
    usage keep working. REST answers ``403 CAPACITY-001`` with ``details.gate
    == "capacity"``, MCP ``capacity_locked``; since v0.126.0 (#1954) a
    context the caller cannot read answers ``context_not_found`` instead.
    Neither a quota nor a plan gate — waiting does not lift it and an
    upgrade is only one way out — so it subclasses neither.

    The overage figures use the server's own key names. A caller who is not
    a member of the locked workspace (reading a public context) gets every
    figure and ``cleanup_url`` as ``None``.

    Attributes:
        gate: Always ``"capacity"``.
        cleanup_url: The web page where the owner cleans up or re-subscribes.
        memory_count: Live memories in the workspace.
        memory_limit: The effective memory limit.
        over_memories: Memories to remove to unlock (0 when within the limit).
        used_bytes: File bytes the owner can still delete.
        storage_limit_bytes: The effective storage limit.
        over_bytes: File bytes to remove to unlock (0 when within the limit).
        details: Every field the server sent, as on :class:`KaguraQuotaError`
            (on MCP also ``help``, which names how much to remove).
    """

    def __init__(
        self,
        message: str,
        *,
        cleanup_url: str | None = None,
        memory_count: int | None = None,
        memory_limit: int | None = None,
        over_memories: int | None = None,
        used_bytes: int | None = None,
        storage_limit_bytes: int | None = None,
        over_bytes: int | None = None,
        details: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.gate = "capacity"
        self.cleanup_url = cleanup_url
        self.memory_count = memory_count
        self.memory_limit = memory_limit
        self.over_memories = over_memories
        self.used_bytes = used_bytes
        self.storage_limit_bytes = storage_limit_bytes
        self.over_bytes = over_bytes
        self.details = dict(details or {})


class KaguraPartialRollbackError(KaguraError):
    """``rollback_sleep_run`` reversed only part of the run (#256).

    Some undo steps failed; the server marks the report ``failed``. The
    steps that succeeded stay reversed, and ``summary`` counts them per
    category, with the failures in ``summary.errors`` — read it to decide
    whether to retry.

    ``rollback_session`` (memory-cloud v0.113.0+, #2044) raises it too when
    items of a session rollback failed: ``report_id`` and ``summary`` are
    then ``None``, and the server's ``summary``, ``totals``, ``remaining``,
    ``rollback_id`` and ``correlation_id`` are in ``details`` (#301).

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
