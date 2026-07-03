"""REST client for workspace member & invitation management (#225).

Owner-key operational tooling: memory-cloud v0.42.0 gates every endpoint
here at workspace-OWNER role for programmatic principals and rejects
OAuth Bearer tokens outright ("Use a workspace-owner API key"). The web
UI (session auth) keeps its own per-endpoint role semantics — this
client only ever sees the programmatic contract.

Construction and credential resolution mirror
:class:`~kagura_memory.files_client.FilesClient`; see that module for the
resolver rationale (#115/#118). Error bodies arrive in the memory-cloud
canonical envelope (``{"error", "message", "details"}``) or the legacy
FastAPI ``{"detail": ...}`` shape — both are handled by
:func:`~kagura_memory._http.extract_detail`.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal
from urllib.parse import quote

import httpx

from ._auth import (
    _SOURCE_LABEL,
    _AuthSource,
    _OAuthAuth,
    _resolve_auth,
    _StaticAuth,
)
from ._http import (
    SDK_VERSION,
    _retry_after_seconds,
    base_url_from_mcp,
    extract_detail,
    validate_https_url,
)
from .auth.credentials import KaguraOAuth
from .exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraQuotaError,
    _exc_message,
)
from .models import MemberAPIKey, WorkspaceInvitation, WorkspaceMember

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


def _validate_workspace_id(workspace_id: str) -> None:
    """Reject non-UUID workspace ids before they reach a URL path."""
    try:
        uuid.UUID(str(workspace_id))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"workspace_id must be a UUID, got {workspace_id!r}") from exc


class WorkspaceClient:
    """REST API client for workspace member / invitation management.

    Requires the workspace OWNER's API key when called programmatically
    (memory-cloud v0.42.0+). OAuth profiles are rejected by the server on
    this surface — resolve a static owner key (``KAGURA_API_KEY`` env or
    ``.kagura.json``) instead.

    All methods may raise:
        KaguraAuthError: Authentication failed (401)
        KaguraConnectionError: Access denied (403 — non-owner key, OAuth
            token, or deployment kill-switch; the message says which) or
            any other HTTP/connection error
        KaguraNotFoundError: Workspace/member/invitation not found (404).
            Also returned for a workspace-scoped key used against a
            different workspace (uniform 404, memory-cloud #963) — a 404
            here does NOT prove the resource is absent.
        KaguraQuotaError: Member quota or rate limit exceeded (429)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://memory.kagura-ai.com",
        timeout: float = 30.0,
        *,
        _oauth: KaguraOAuth | None = None,
        _auth_source: _AuthSource | None = None,
        _workspace_id_hint: str | None = None,
    ) -> None:
        """Initialize WorkspaceClient with a static API key.

        For credential resolution (the auto chain env → ``~/.kagura/
        credentials.json`` → ``.kagura.json``), use :meth:`from_mcp_url`.

        Args:
            api_key: Kagura API key (Bearer token). Required unless
                ``_oauth`` is supplied (see ``from_mcp_url``).
            base_url: REST API base URL (without path).
            timeout: API request timeout in seconds.
            _oauth: Private. ``from_mcp_url`` passes a ``KaguraOAuth``
                instance when the resolver picked an OAuth profile. Kept
                for construction parity — the server rejects OAuth on
                this surface with an actionable 403, which is a better
                operator experience than failing client-side with a
                generic credential error.
            _auth_source: Private. Provenance tag from ``_resolve_auth``;
                drives the actionable 403 hint (issue #115).
            _workspace_id_hint: Private. Workspace UUID associated with
                the credential source — 403 hint display only, never sent
                on the wire.

        Raises:
            ValueError: If neither ``api_key`` nor ``_oauth`` is given.
        """
        if api_key is None and _oauth is None:
            raise ValueError(
                "WorkspaceClient requires api_key, or use "
                "WorkspaceClient.from_mcp_url(...) to resolve credentials from "
                "environment, OAuth profile, or .kagura.json."
            )

        stripped_url = base_url.rstrip("/")
        validate_https_url(stripped_url, label="Base URL")
        self.base_url = stripped_url

        if _oauth is not None:
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={"User-Agent": f"kagura-memory-sdk/{SDK_VERSION}"},
                auth=_oauth,
            )
        else:
            # Bake the bearer header once; api_key is not stored as an
            # instance attribute (python.md "Never store API keys ...").
            self._client = httpx.AsyncClient(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": f"kagura-memory-sdk/{SDK_VERSION}",
                },
            )
        self._auth_source: _AuthSource | None = _auth_source
        self._workspace_id_hint: str | None = _workspace_id_hint

    @classmethod
    def from_mcp_url(
        cls,
        api_key: str | None = None,
        mcp_url: str | None = None,
        timeout: float = 30.0,
        *,
        profile: str | None = None,
    ) -> WorkspaceClient:
        """Create WorkspaceClient by resolving credentials from the SDK chain.

        Same precedence as :meth:`FilesClient.from_mcp_url`: explicit
        ``api_key`` > ``KAGURA_API_KEY`` env > OAuth profile >
        ``.kagura.json``. Note that the server accepts only static owner
        API keys on this surface; an OAuth profile will connect but every
        call returns an actionable 403.

        Args:
            api_key: Explicit Kagura API key. Skips the resolution chain.
            mcp_url: Explicit MCP URL. When omitted, the resolved
                credential source's stored URL is used.
            timeout: API request timeout in seconds.
            profile: Named OAuth profile to load.
        """
        resolved = _resolve_auth(api_key=api_key, mcp_url=mcp_url, profile=profile)
        return cls._from_resolved_auth(resolved, timeout=timeout)

    @classmethod
    def _from_resolved_auth(
        cls,
        resolved: _StaticAuth | _OAuthAuth,
        *,
        timeout: float = 30.0,
        workspace_id_hint: str | None = None,
    ) -> WorkspaceClient:
        """Construct from a pre-resolved auth — internal CLI helper.

        Mirrors ``FilesClient._from_resolved_auth`` so the CLI can resolve
        once and pair api_key with its same-source workspace (#115).
        """
        base_url = base_url_from_mcp(resolved.mcp_url.rstrip("/"))
        if isinstance(resolved, _StaticAuth):
            return cls(
                api_key=resolved.api_key,
                base_url=base_url,
                timeout=timeout,
                _auth_source=resolved.source,
                _workspace_id_hint=workspace_id_hint,
            )
        return cls(
            base_url=base_url,
            timeout=timeout,
            _oauth=resolved.oauth,
            _auth_source="oauth",
            _workspace_id_hint=resolved.workspace_id,
        )

    async def __aenter__(self) -> WorkspaceClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # -------------------------------------------------------------------
    # Members
    # -------------------------------------------------------------------

    async def list_members(self, workspace_id: str) -> list[WorkspaceMember]:
        """List workspace members (owner key required).

        Rows are ordered owner→viewer then by join date and include
        ``user_name``/``user_email`` when the member has logged in.
        """
        _validate_workspace_id(workspace_id)
        resp = await self._request("GET", f"/api/v1/workspaces/{workspace_id}/members")
        return [WorkspaceMember.model_validate(row) for row in resp.json()]

    async def add_member(
        self, workspace_id: str, user_id: str, role: str = "member"
    ) -> WorkspaceMember:
        """Add an already-registered user to the workspace.

        The server does NOT validate that ``user_id`` exists (v0.42.0) — a
        typo creates a dangling membership row that lists with null
        name/email. Prefer :meth:`create_invitation` for onboarding;
        use this only with a user_id copied from a trusted source.
        Duplicate members are rejected with 422.
        """
        _validate_workspace_id(workspace_id)
        self._validate_role(role)
        resp = await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/members",
            json={"user_id": user_id, "role": role},
        )
        return WorkspaceMember.model_validate(resp.json())

    async def update_member_role(
        self, workspace_id: str, user_id: str, role: str
    ) -> WorkspaceMember:
        """Change a member's role (member/admin/viewer only)."""
        _validate_workspace_id(workspace_id)
        self._validate_role(role)
        resp = await self._request(
            "PUT",
            f"/api/v1/workspaces/{workspace_id}/members/{quote(user_id, safe='')}",
            json={"role": role},
        )
        return WorkspaceMember.model_validate(resp.json())

    async def remove_member(self, workspace_id: str, user_id: str) -> None:
        """Remove a member from the workspace (server returns 204)."""
        _validate_workspace_id(workspace_id)
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/members/{quote(user_id, safe='')}",
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
                member/viewer invitations, ignored for admin.
            expires_in_days: One of 7/30/90/365, or None = never expires.
        """
        _validate_workspace_id(workspace_id)
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
            body["allowed_context_ids"] = allowed_context_ids
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        resp = await self._request(
            "POST", f"/api/v1/workspaces/{workspace_id}/invitations", json=body
        )
        return WorkspaceInvitation.model_validate(resp.json())

    async def list_invitations(
        self, workspace_id: str, *, include_accepted: bool = False
    ) -> list[WorkspaceInvitation]:
        """List invitations. ``token``/``invitation_url`` arrive as null
        for programmatic callers (server-side token hygiene, #1164)."""
        _validate_workspace_id(workspace_id)
        params = {"include_accepted": "true"} if include_accepted else None
        resp = await self._request(
            "GET", f"/api/v1/workspaces/{workspace_id}/invitations", params=params
        )
        return [WorkspaceInvitation.model_validate(row) for row in resp.json()]

    async def revoke_invitation(self, workspace_id: str, invitation_id: int) -> None:
        """Revoke a pending invitation (server returns 200 {"success": true})."""
        _validate_workspace_id(workspace_id)
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/invitations/{int(invitation_id)}",
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
        """
        _validate_workspace_id(workspace_id)
        if not 1 <= int(expires_days) <= 3650:
            raise ValueError(f"expires_days must be 1-3650, got {expires_days!r}")
        resp = await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/members/"
            f"{quote(user_id, safe='')}/credentials/api-keys",
            json={"name": name, "expires_days": int(expires_days)},
        )
        return MemberAPIKey.model_validate(resp.json())

    async def list_member_keys(self, workspace_id: str, user_id: str) -> list[MemberAPIKey]:
        """List a member's API keys — metadata only.

        The server wraps the rows in a ``MemberCredentialsResponse``
        envelope (``api_keys`` + ``target_user_role``) and always nulls
        ``plaintext_key`` for programmatic callers.
        """
        _validate_workspace_id(workspace_id)
        resp = await self._request(
            "GET",
            f"/api/v1/workspaces/{workspace_id}/members/{quote(user_id, safe='')}/credentials",
        )
        payload = resp.json()
        return [MemberAPIKey.model_validate(row) for row in payload.get("api_keys", [])]

    async def revoke_member_key(self, workspace_id: str, user_id: str, key_id: int) -> None:
        """Revoke a member's API key.

        Owner-provisioned revocations are SOFT server-side (``revoked_at``
        set, row retained for forensics); success is 200 with a status
        body, and an already-revoked key surfaces as a uniform 404.
        """
        _validate_workspace_id(workspace_id)
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/members/"
            f"{quote(user_id, safe='')}/credentials/api-keys/{int(key_id)}",
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
        if server_detail and server_detail != _UNIFORM_403:
            return server_detail
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

    async def _request(
        self,
        method: Literal["GET", "POST", "PUT", "DELETE"],
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make an authenticated request with standard error mapping."""
        url = f"{self.base_url}{path}"
        try:
            response = await self._client.request(method, url, json=json, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = extract_detail(e.response)
            if status == 401:
                raise KaguraAuthError("Authentication failed. Check your API key.") from e
            if status == 403:
                raise KaguraConnectionError(self._format_403(detail)) from e
            if status == 404:
                raise KaguraNotFoundError(detail or "Not found") from e
            if status == 429:
                # Invite-create quota exhaustion ("Member limit reached ...")
                # and generic rate limiting both surface as 429 — keep the
                # server message, it names the cause and the fix.
                raise KaguraQuotaError(
                    detail or "Quota exceeded. Try again later.",
                    retry_after=_retry_after_seconds(e.response),
                ) from e
            msg = f"HTTP {status}: {detail}" if detail else f"HTTP {status}"
            raise KaguraConnectionError(msg) from e
        except httpx.RequestError as e:
            raise KaguraConnectionError(f"Connection failed: {_exc_message(e)}") from e
