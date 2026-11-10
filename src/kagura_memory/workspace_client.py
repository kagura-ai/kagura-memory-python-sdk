"""REST client for workspace member & invitation management (#225).

Owner-key operational tooling: memory-cloud v0.42.0 gates every endpoint
here at workspace-OWNER role for programmatic principals and rejects
OAuth Bearer tokens outright ("Use a workspace-owner API key"). The web
UI (session auth) keeps its own per-endpoint role semantics — this
client only ever sees the programmatic contract.

Construction, credential resolution, lifecycle, and the base error
mapping live in :class:`~kagura_memory._rest_base.KaguraRestClient`
(#229); this module keeps only the workspace-specific wire contract:
the owner-key 403 hint and the detail-carrying 429 quota mapping.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import ValidationError

from ._auth import _SOURCE_LABEL
from ._http import (
    ErrorBody,
    _retry_after_seconds,
    normalize_uuid,
    path_segment,
    public_id_segment,
    sanitize_server_detail,
)
from ._rest_base import KaguraRestClient
from .exceptions import (
    KaguraConnectionError,
    KaguraContextAccessListError,
    KaguraError,
    KaguraQuotaError,
    KaguraResponseError,
)
from .models import (
    ContextMember,
    MemberAPIKey,
    MemberContextAccess,
    MemberContextAccessUpdate,
    OperationHistoryOutcome,
    OperationHistoryPage,
    OperationHistoryTargetType,
    WorkspaceInvitation,
    WorkspaceMember,
    _validate_int_range,
)

VALID_ASSIGNABLE_ROLES = ("member", "admin", "viewer")
# "owner" is excluded: the server 422s programmatic role=owner on member
# add/set-role (memory-cloud#1164 — "use the ownership transfer flow") and
# rejects owner invitations for every principal (memory-cloud#1166).

VALID_INVITE_EXPIRES = (7, 30, 90, 365)
# Server-side preset list: any other value in the pydantic-valid 1-365 range
# still 400s at the service layer, so fail fast client-side. None = never.

_UNIFORM_403 = "Insufficient permissions"
# The deliberately uniform authorization denial (CWE-639). Every other 403 on
# this surface carries a purpose-built message (OAuth rejection, deployment
# kill-switch, plan gate, self-role-change, key/workspace mismatch) that is
# already actionable and must pass through untouched.


def _normalize_workspace_id(workspace_id: str) -> str:
    """Canonicalize before URL interpolation — see :func:`normalize_uuid`."""
    return normalize_uuid(workspace_id, label="workspace_id")


def _user_id_segment(user_id: str) -> str:
    """``user_id`` as one path segment; ``.``, ``..`` and empty are refused (#285).

    Percent-encoding leaves ``.`` and ``..`` as they are, so
    ``remove_member(ws, "..")`` would send ``DELETE /api/v1/workspaces/<ws>``,
    the workspace itself. See :func:`path_segment`.
    """
    return path_segment(user_id, label="user_id")


def _require_int(value: object, label: str) -> int:
    """Strictly require an int — no bool, no float truncation, no str parse.

    ``int(7.9)`` would silently target a DIFFERENT value, and ``int("7a")``
    raises a bare ValueError with no context; both are worth failing loudly
    instead. Resource ids use :func:`public_id_segment` (int or str).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value


def _context_ids(ids: list[str]) -> list[str]:
    if not isinstance(ids, list):
        raise ValueError("allowed_context_ids must be a list of UUIDs")
    return [normalize_uuid(c, label="allowed_context_ids") for c in ids]


# The operation-history route's query bounds (memory-cloud v0.155.0,
# api/routes/operation_history.py ``Query(max_length=...)``, on the raw value).
_HISTORY_QUERY_MAX = 120
_HISTORY_ACTOR_MAX = 255
_HISTORY_OPERATION_MAX = 40
_HISTORY_CURSOR_MAX = 512
_HISTORY_LIMIT_MAX = 100


def _history_text(name: str, value: object, cap: int | None = None) -> str | None:
    """A text filter as sent: ``None`` when unset, else the ``str`` unchanged.

    Only ``None`` is unset: a blank value is sent, and the server decides what
    it means (no filter for ``query`` / ``actor``, a 422 for ``operation``).
    The value is never echoed.

    Raises:
        ValueError: If ``value`` is not a ``str`` or is over ``cap`` characters.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    if cap is not None and len(value) > cap:
        raise ValueError(f"{name} must be at most {cap} characters, got {len(value)}")
    return value


def _history_cursor(value: object) -> str | None:
    """The opaque ``cursor`` as sent; a blank one is refused (never server-issued).

    Sending ``""`` would silently restart at page one. The value is never echoed.

    Raises:
        ValueError: If ``cursor`` is not a ``str``, is blank or is over 512 characters.
    """
    cursor = _history_text("cursor", value, _HISTORY_CURSOR_MAX)
    if cursor is not None and not cursor.strip():
        raise ValueError("cursor must not be blank: pass a next_cursor the server returned")
    return cursor


def _history_time(name: str, value: object) -> datetime | None:
    """A ``since`` / ``until`` filter: a timezone-aware ``datetime`` or ``None``.

    A naive value is refused, as the server refuses it; it is not taken as UTC.

    Raises:
        ValueError: If ``value`` is not a ``datetime`` or has no timezone.
    """
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware (the server refuses a naive time)")
    return value


def _history_time_param(value: datetime | None) -> str | None:
    """The same instant in UTC: an offset with seconds would not parse on the server."""
    return None if value is None else value.astimezone(UTC).isoformat()


def _access_path(workspace_id: str, user_id: str) -> str:
    ws = _normalize_workspace_id(workspace_id)
    return f"/api/v1/workspaces/{ws}/members/{_user_id_segment(user_id)}/context-access"


class WorkspaceClient(KaguraRestClient):
    """REST API client for workspace member / invitation management.

    Requires the workspace OWNER's API key when called programmatically
    (memory-cloud v0.42.0+). OAuth profiles are rejected by the server on
    this surface — resolve a static owner key (``KAGURA_API_KEY`` env or
    ``.kagura.json``) instead.

    Context access (memory-cloud v0.126.0+): an owner or admin reaches every
    shared context of the workspace. A member or viewer reaches exactly the
    contexts its ``allowed_context_ids`` list names. With no list, a viewer
    reaches every shared context and a member **none**, writes included.
    Older servers refused only a member's reads there. A private context
    stays its creator's (or a linked account's), whatever the role or the
    list. A context outside the list answers not-found (REST 404, MCP
    ``context_not_found``, :class:`KaguraNotFoundError`) where some paths
    answered 403. A workspace viewer is read-only everywhere, including a
    private context it created. A role change keeps the list, except that
    a ``member`` with no list who becomes a ``viewer`` gets an empty one,
    so the change never widens its reach.

    On memory-cloud v0.144.0+, the owner key can set the list at member add
    and read or replace it via the member /context-access endpoint.
    The separate context-member POST requires a signed-in web session, and so
    does :meth:`list_operation_history` (v0.154.0+, an owner or admin).

    All methods may raise:
        KaguraAuthError: Authentication failed (401)
        KaguraConnectionError: Access denied (403 — non-owner key, OAuth
            token, or deployment kill-switch; the message says which) or
            any other HTTP/connection error
        KaguraNotFoundError: Workspace/member/invitation not found (404).
            Also returned for a workspace-scoped key used against a
            different workspace (uniform 404, memory-cloud #963) — a 404
            here does NOT prove the resource is absent.
        KaguraFeatureNotAvailableError: The plan lacks team invitations
            (``FEAT-001``, server v0.75.0+; older servers answer a plain
            403, raised as KaguraConnectionError), or — on every route,
            server v0.100.0+ (memory-cloud#1943) — the caller is a member
            without a seat after the plan dropped below the workspace's
            members (``feature == "members"``, with ``limit`` / ``current``
            in ``details``)
        KaguraQuotaError: Member seat cap (``quota_type="members"`` on
            server v0.75.0+) or rate limit exceeded (429)
    """

    # -------------------------------------------------------------------
    # Members
    # -------------------------------------------------------------------

    async def list_members(self, workspace_id: str) -> list[WorkspaceMember]:
        """List workspace members (owner key required).

        Rows include ``user_name``/``user_email`` when the member has logged
        in. Since memory-cloud v0.100.0 (#1943) they come in seat order —
        the owner first, then earliest joined — and the members past the
        seats the plan carries have ``seat_paused`` set. Older servers sorted
        them by the role name as a string.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        resp = await self._request("GET", f"/api/v1/workspaces/{workspace_id}/members")
        return [
            self._parse(WorkspaceMember, row, "list_members")
            for row in self._expect_list(resp, "list_members")
        ]

    async def add_member(
        self,
        workspace_id: str,
        user_id: str,
        role: str = "member",
        *,
        allowed_context_ids: list[str] | None = None,
    ) -> WorkspaceMember:
        """Add an already-registered user to the workspace.

        The server does NOT validate that ``user_id`` exists (v0.42.0) — a
        typo creates a dangling membership row that lists with null
        name/email. Prefer :meth:`create_invitation` for onboarding;
        use this only with a user_id copied from a trusted source.
        Duplicate members are rejected with 422. ``.``, ``..`` and an
        empty id are refused (``ValueError``): no other method could
        address such a member afterwards.

        Since memory-cloud v0.99.0 (#1939) this checks the plan like
        :meth:`create_invitation`: a plan without the ``team_invitations``
        feature raises :class:`KaguraFeatureNotAvailableError` and a full
        seat cap :class:`KaguraQuotaError` (``quota_type="members"``).

        From v0.144.0, allowed_context_ids can be supplied at add time. An empty
        list grants no contexts. Omission uses the role default: on v0.126.0+
        members reach no context and viewers reach every shared context.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        _user_id_segment(user_id)
        self._validate_role(role)
        body: dict[str, Any] = {"user_id": user_id, "role": role}
        if allowed_context_ids is not None:
            body["allowed_context_ids"] = _context_ids(allowed_context_ids)
        resp = await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/members",
            json=body,
        )
        return self._parse(WorkspaceMember, self._json(resp), "add_member")

    async def get_member_context_access(
        self, workspace_id: str, user_id: str
    ) -> MemberContextAccess:
        """Read the stored list and assignable contexts (owner key, v0.144.0+)."""
        resp = await self._request("GET", _access_path(workspace_id, user_id))
        return self._parse(MemberContextAccess, self._json(resp), "get_member_context_access")

    async def set_member_context_access(
        self, workspace_id: str, user_id: str, allowed_context_ids: list[str] | None
    ) -> MemberContextAccessUpdate:
        """Replace the list. None restores the role default, [] grants no contexts."""
        ids = None if allowed_context_ids is None else _context_ids(allowed_context_ids)
        resp = await self._request(
            "PUT", _access_path(workspace_id, user_id), json={"allowed_context_ids": ids}
        )
        return self._parse(MemberContextAccessUpdate, self._json(resp), "set_member_context_access")

    @classmethod
    def from_web_session(
        cls, *, base_url: str, session_cookie: str, timeout: float = 30.0
    ) -> WorkspaceClient:
        """Use the kagura_session cookie from a signed-in session on this server."""
        return cls(base_url=base_url, timeout=timeout, _session_cookie=session_cookie)

    async def add_context_member(
        self, context_id: str, user_id: str, role: str, *, extend_access_list: bool | None = None
    ) -> ContextMember:
        """Add a shared-context member. Extending the list requires workspace admin/owner."""
        if not self._uses_web_session:
            raise ValueError("add_context_member requires WorkspaceClient.from_web_session()")
        context = normalize_uuid(context_id, label="context_id")
        _user_id_segment(user_id)
        if role not in ("owner", "editor", "viewer"):
            raise ValueError("role must be owner, editor, or viewer")
        body: dict[str, Any] = {"user_id": user_id, "role": role}
        if extend_access_list is not None:
            body["extend_access_list"] = extend_access_list
        resp = await self._request("POST", f"/api/v1/contexts/{context}/members", json=body)
        return self._parse(ContextMember, self._json(resp), "add_context_member")

    # -------------------------------------------------------------------
    # Operation history (#332, memory-cloud v0.154.0+, web session only)
    # -------------------------------------------------------------------

    async def list_operation_history(
        self,
        workspace_id: str,
        *,
        query: str | None = None,
        context_id: str | None = None,
        actor: str | None = None,
        agent_id: str | None = None,
        operation: str | None = None,
        target_type: OperationHistoryTargetType | None = None,
        outcome: OperationHistoryOutcome | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> OperationHistoryPage:
        """Read one page of the workspace's operation history (memory-cloud v0.154.0+).

        ``GET /api/v1/workspaces/{workspace_id}/operation-history`` takes a
        signed-in web session only (:meth:`from_web_session`): an API key,
        an OAuth token or a resource token is refused here before any
        request. The caller must be an owner or admin of the workspace, and
        its plan must include the ``operation_history`` feature (the Pro and
        Max tiers by default); both are rechecked on every page.

        The history is recorded by the server from when the feature was
        deployed or the workspace moved to an entitled plan, with no
        backfill (Free and Starter record nothing), and kept for
        ``retention_days`` (90 plus the workspace's add-on). A private
        context's events are visible only to its creator. A shared
        credential's owner is not asserted to be the physical caller. It is
        not an immutable audit ledger, and there is no export.

        One page per call: pass ``cursor=page.next_cursor`` with the same
        filters until it is empty (``None``; stop on any falsy value). Only
        ``None`` filters are left out; every other value is sent as given
        after the local checks, so a blank ``query`` / ``actor`` reaches the
        server as no filter and a blank ``operation`` is the server's 422.
        ``since`` / ``until`` are sent as the same instant in UTC.

        Args:
            workspace_id: Workspace UUID.
            query: Text search over actor names/ids, target, context, agent,
                credential and client ids, operation names and changed-field
                names; at most 120 characters (the server trims it).
            context_id: Only this context's events (UUID).
            actor: Exactly this actor user id (the server trims it; a name
                does not match here, use ``query``), at most 255 characters.
            agent_id: Only this agent's events (UUID).
            operation: At most 40 characters. The server refuses a value
                outside its set, at v0.155.0: ``memory.created``,
                ``memory.updated``, ``memory.deleted``, ``memory.restored``,
                ``context.created``, ``context.updated``, ``context.deleted``,
                ``context.restored``, ``member.invited``, ``member.added``,
                ``member.removed``, ``member.role_changed``,
                ``member.access_changed``, ``credential.issued``,
                ``credential.updated``, ``credential.revoked``,
                ``integration.created``, ``integration.updated``,
                ``integration.deleted``, ``sleep.completed``,
                ``sleep.failed`` and ``sleep.undo``.
            target_type: ``memory`` | ``context`` | ``member`` |
                ``credential`` | ``integration`` | ``sleep``; sent as given.
            outcome: ``success`` | ``failure`` | ``denied``; sent as given.
            since: Events at or after this timezone-aware time.
            until: Events at or before this timezone-aware time.
            limit: Events per page, 1-100 (server default 30).
            cursor: ``next_cursor`` of the previous page, sent back as is
                (opaque, at most 512 characters; a blank one is refused).

        Returns:
            :class:`~kagura_memory.models.OperationHistoryPage`.

        Raises:
            ValueError: Before any request: the client is not a web-session
                client; ``workspace_id`` / ``context_id`` / ``agent_id`` is
                not a UUID; a text filter (``target_type`` and ``outcome``
                included) is not a ``str`` or is over its cap; ``cursor``
                is blank; ``limit`` is not an ``int`` in 1-100; ``since`` /
                ``until`` is not a timezone-aware ``datetime``; or ``since``
                is after ``until``.
            KaguraAuthError: The web session expired or is invalid (401).
            KaguraFeatureNotAvailableError: The workspace's plan lacks
                ``operation_history`` (``FEAT-001``, ``feature ==
                "operation_history"``).
            KaguraConnectionError: The caller is not an owner or admin of
                the workspace, or the workspace is unknown or deleted (the
                same uniform 403 ``AUTH-101`` "Insufficient permissions"),
                a filter the server refused (422, e.g. an unknown or blank
                ``operation`` or an invalid cursor), or another HTTP or
                network failure.
            KaguraNotFoundError: The server has no operation-history route
                (404) — it is older than v0.154.0. On v0.154.0+ an unknown
                workspace gets the uniform 403 above, never a 404.
            KaguraResponseError: The page does not match
                :class:`~kagura_memory.models.OperationHistoryPage`.
        """
        if not self._uses_web_session:
            raise ValueError("list_operation_history requires WorkspaceClient.from_web_session()")
        workspace = _normalize_workspace_id(workspace_id)
        if limit is not None:
            _validate_int_range("limit", limit, 1, _HISTORY_LIMIT_MAX)
        since = _history_time("since", since)
        until = _history_time("until", until)
        if since is not None and until is not None and since > until:
            raise ValueError("since must not be after until")
        params: dict[str, Any] = {
            "query": _history_text("query", query, _HISTORY_QUERY_MAX),
            "context_id": (
                None if context_id is None else normalize_uuid(context_id, label="context_id")
            ),
            "actor": _history_text("actor", actor, _HISTORY_ACTOR_MAX),
            "agent_id": None if agent_id is None else normalize_uuid(agent_id, label="agent_id"),
            "operation": _history_text("operation", operation, _HISTORY_OPERATION_MAX),
            "target_type": _history_text("target_type", target_type),
            "outcome": _history_text("outcome", outcome),
            "since": _history_time_param(since),
            "until": _history_time_param(until),
            "limit": limit,
            "cursor": _history_cursor(cursor),
        }
        resp = await self._request(
            "GET",
            f"/api/v1/workspaces/{workspace}/operation-history",
            params={k: v for k, v in params.items() if v is not None},
        )
        return self._parse(OperationHistoryPage, self._json(resp), "list_operation_history")

    async def update_member_role(
        self, workspace_id: str, user_id: str, role: str
    ) -> WorkspaceMember:
        """Change a member's role (member/admin/viewer only).

        The member's context access list is kept (memory-cloud v0.126.0+),
        except that a ``member`` with no list who becomes a ``viewer`` gets
        an empty one: a viewer with no list would reach every shared
        context, so the change never widens reach. The list applies to the
        member and viewer roles; an admin reaches every shared context
        whatever it says.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        self._validate_role(role)
        resp = await self._request(
            "PUT",
            f"/api/v1/workspaces/{workspace_id}/members/{_user_id_segment(user_id)}",
            json={"role": role},
        )
        return self._parse(WorkspaceMember, self._json(resp), "update_member_role")

    async def remove_member(self, workspace_id: str, user_id: str) -> None:
        """Remove a member from the workspace (server returns 204).

        The server hands the member's memories, contexts and resource tokens
        to the workspace owner and deletes their credentials and context
        memberships. Since memory-cloud v0.129.0 (#2352) it also hands their
        live connector write keys in this workspace to the owner, so the
        connector keeps working. Since v0.139.0 (#2373 / #2349) their
        agent-bound keys are deleted, an active agent left with no live key
        is set ``suspended``, and an agent they registered is re-owned to
        the workspace owner.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/members/{_user_id_segment(user_id)}",
        )

    # -------------------------------------------------------------------
    # Invitations
    # -------------------------------------------------------------------

    async def create_invitation(
        self,
        workspace_id: str,
        email: str,
        role: str = "member",
        *,
        allowed_context_ids: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> WorkspaceInvitation:
        """Invite a not-yet-registered user by email.

        The returned invitation carries ``token``/``invitation_url`` —
        the only response that ever exposes them to programmatic callers;
        treat them as join credentials.

        Args:
            workspace_id: Target workspace UUID.
            email: Invitee email (must match their Google account).
            role: ``member`` | ``admin`` | ``viewer``.
            allowed_context_ids: Context grant — required (min 1) for
                member/viewer invitations, ignored for admin. Each must be
                a UUID (``ValueError`` otherwise); sent in canonical form.
                It becomes the member's context access list: since
                memory-cloud v0.126.0 a member or viewer reaches exactly
                these contexts on every path.
            expires_in_days: One of 7/30/90/365, or None = never expires.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        self._validate_role(role)
        if role in ("member", "viewer") and not allowed_context_ids:
            raise ValueError(
                "allowed_context_ids is required (min 1) for member/viewer "
                "invitations — the server rejects them without a context grant."
            )
        if expires_in_days is not None and expires_in_days not in VALID_INVITE_EXPIRES:
            raise ValueError(
                f"expires_in_days must be one of {VALID_INVITE_EXPIRES} or None "
                "(server-side preset list)."
            )
        body: dict[str, Any] = {"email": email, "role": role}
        if allowed_context_ids is not None:
            # Checked locally on every server: memory-cloud before v0.80.0
            # answered a non-UUID here with an HTTP 500 (#285), and from
            # v0.80.0 (#1693) answers 422 VAL-001.
            body["allowed_context_ids"] = _context_ids(allowed_context_ids)
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        resp = await self._request(
            "POST", f"/api/v1/workspaces/{workspace_id}/invitations", json=body
        )
        return self._parse(WorkspaceInvitation, self._json(resp), "create_invitation")

    async def list_invitations(
        self, workspace_id: str, *, include_accepted: bool = False
    ) -> list[WorkspaceInvitation]:
        """List invitations. ``token``/``invitation_url`` arrive as null
        for programmatic callers (server-side token hygiene, #1164)."""
        workspace_id = _normalize_workspace_id(workspace_id)
        params = {"include_accepted": "true"} if include_accepted else None
        resp = await self._request(
            "GET", f"/api/v1/workspaces/{workspace_id}/invitations", params=params
        )
        return [
            self._parse(WorkspaceInvitation, row, "list_invitations")
            for row in self._expect_list(resp, "list_invitations")
        ]

    async def revoke_invitation(self, workspace_id: str, invitation_id: int | str) -> None:
        """Revoke a pending invitation (server returns 200 {"success": true}).

        ``invitation_id`` is an int on servers before v0.89.0 and an opaque
        ``winv_…`` string from v0.89.0 on — pass whatever ``list_invitations``
        returned.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/invitations/"
            f"{public_id_segment(invitation_id, label='invitation_id')}",
        )

    # -------------------------------------------------------------------
    # Member API keys (#201, memory-cloud#1165)
    # -------------------------------------------------------------------

    async def mint_member_key(
        self, workspace_id: str, user_id: str, name: str, expires_days: int
    ) -> MemberAPIKey:
        """Mint an API key for ANOTHER member (owner key required).

        Privilege-downgrade provisioning only: the server 403s
        self-targets ("An owner key cannot mint keys for itself") and
        owner/admin targets — mint for member/viewer service identities.
        ``expires_days`` is required by the server for owner-provisioned
        mints (never-expiring CI keys are not allowed). The returned
        ``plaintext_key`` is shown exactly once — owner-provisioned keys
        are force-hidden at creation, so no later call returns it.
        Since memory-cloud v0.83.0 (#1752) the target member gets a
        security email, at a verified address, for every key minted for
        them; it never carries the key.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        expires_days = _require_int(expires_days, "expires_days")
        if not 1 <= expires_days <= 3650:
            raise ValueError(f"expires_days must be 1-3650, got {expires_days!r}")
        resp = await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/members/"
            f"{_user_id_segment(user_id)}/credentials/api-keys",
            json={"name": name, "expires_days": expires_days},
        )
        payload = self._json(resp)
        try:
            return MemberAPIKey.model_validate(payload)
        except ValidationError as exc:
            # The key already exists server-side and is force-hidden — a shape
            # mismatch must not swallow the ONE chance to see the plaintext,
            # so this path writes its own message instead of using _parse (the
            # one KaguraResponseError whose message carries a payload value).
            operation = self._operation("mint_member_key")
            plaintext = payload.get("plaintext_key") if isinstance(payload, dict) else None
            if isinstance(plaintext, str) and plaintext:
                raise KaguraResponseError(
                    f"{operation}: server returned an unexpected mint response shape, "
                    f"but the key WAS created. Save the plaintext now: {plaintext}",
                    operation=operation,
                ) from exc
            raise KaguraResponseError(
                f"{operation}: server returned an unexpected mint response shape; "
                "the key may have been created without displaying its plaintext — check "
                "`kagura auth list-keys` and revoke/re-mint if present.",
                operation=operation,
            ) from exc

    async def list_member_keys(self, workspace_id: str, user_id: str) -> list[MemberAPIKey]:
        """List a member's API keys — metadata only.

        The server wraps the rows in a ``MemberCredentialsResponse``
        envelope (``api_keys`` + ``target_user_role``) and always nulls
        ``plaintext_key`` for programmatic callers.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        resp = await self._request(
            "GET",
            f"/api/v1/workspaces/{workspace_id}/members/{_user_id_segment(user_id)}/credentials",
        )
        return [
            self._parse(MemberAPIKey, row, "list_member_keys")
            for row in self._expect_wrapped_list(resp, "api_keys", "list_member_keys")
        ]

    async def revoke_member_key(self, workspace_id: str, user_id: str, key_id: int | str) -> None:
        """Revoke a member's API key.

        Owner-provisioned revocations are SOFT server-side (``revoked_at``
        set, row retained for forensics); success is 200 with a status
        body, and an already-revoked key surfaces as a uniform 404.
        ``key_id`` is an int before server v0.89.0 and an opaque ``akey_…``
        string from v0.89.0 on.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/members/"
            f"{_user_id_segment(user_id)}/credentials/api-keys/"
            f"{public_id_segment(key_id, label='key_id')}",
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _validate_role(self, role: str) -> None:
        if role not in VALID_ASSIGNABLE_ROLES:
            raise ValueError(
                f"role must be one of {VALID_ASSIGNABLE_ROLES}, got {role!r}. "
                "Assigning 'owner' is not supported programmatically — the "
                "server directs owner changes to the ownership transfer flow."
            )

    def _format_403(self, server_detail: str) -> str:
        """Build the 403 message, appending the owner-key hint only when useful.

        v0.42.0 sends purpose-built 403 messages on this surface (OAuth
        rejection, deployment kill-switch, plan gate, self-role-change,
        key/workspace mismatch) — those pass through untouched. Only the
        deliberately uniform denial gets the owner-key hint, because that
        is the one a non-owner static key actually hits.
        """
        safe_detail = sanitize_server_detail(server_detail)
        if safe_detail and safe_detail != _UNIFORM_403:
            return safe_detail
        parts = [
            "Access denied (HTTP 403): workspace member/invitation/credential "
            "management requires the workspace OWNER's API key when called "
            "programmatically (OAuth tokens are not accepted)."
        ]
        if self._auth_source is not None:
            label = _SOURCE_LABEL.get(self._auth_source, str(self._auth_source))
            hint = f"credential source: {label}"
            if self._workspace_id_hint:
                hint += f" (workspace={self._workspace_id_hint[:8]}…)"
            parts.append(hint + " — is this key the workspace owner's?")
        return " ".join(parts)

    # ---- KaguraRestClient hooks -----------------------------------------

    def _error_403(
        self,
        e: httpx.HTTPStatusError,
        body: ErrorBody,
        *,
        request_json: dict[str, Any] | None,
        request_params: dict[str, Any] | None,
    ) -> KaguraError:
        if self._uses_web_session:
            return self._generic_error(e, body)
        return KaguraConnectionError(self._format_403(body.detail))

    def _generic_error(self, e: httpx.HTTPStatusError, body: ErrorBody) -> KaguraError:
        if e.response.status_code == 409 and body.envelope is not None:
            code, message, details = body.envelope
            if code == "RES-002" and details.get("reason") == "context_not_in_access_list":
                return KaguraContextAccessListError(sanitize_server_detail(message))
        return super()._generic_error(e, body)

    def _error_429(self, e: httpx.HTTPStatusError, body: ErrorBody) -> KaguraError:
        # Invite-create quota exhaustion ("Member limit reached ...") and
        # generic rate limiting both surface as 429 — keep the server
        # message, it names the cause and the fix.
        return KaguraQuotaError(
            body.detail or "Quota exceeded. Try again later.",
            retry_after=_retry_after_seconds(e.response),
        )
