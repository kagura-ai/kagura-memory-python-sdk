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

from typing import Any
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from ._auth import _SOURCE_LABEL
from ._http import _retry_after_seconds, extract_detail, normalize_uuid, sanitize_server_detail
from ._rest_base import KaguraRestClient
from .exceptions import KaguraConnectionError, KaguraError, KaguraQuotaError
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


def _normalize_workspace_id(workspace_id: str) -> str:
    """Canonicalize before URL interpolation — see :func:`normalize_uuid`."""
    return normalize_uuid(workspace_id, label="workspace_id")


def _require_int(value: object, label: str) -> int:
    """Strictly require an int — no bool, no float truncation, no str parse.

    ``int(7.9)`` would silently target a DIFFERENT resource id on a
    destructive endpoint, and ``int("7a")`` raises a bare ValueError with
    no context; both are worth failing loudly instead.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    return value


class WorkspaceClient(KaguraRestClient):
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

    # -------------------------------------------------------------------
    # Members
    # -------------------------------------------------------------------

    async def list_members(self, workspace_id: str) -> list[WorkspaceMember]:
        """List workspace members (owner key required).

        Rows are ordered owner→viewer then by join date and include
        ``user_name``/``user_email`` when the member has logged in.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        resp = await self._request("GET", f"/api/v1/workspaces/{workspace_id}/members")
        return [WorkspaceMember.model_validate(row) for row in self._expect_list(resp)]

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
        workspace_id = _normalize_workspace_id(workspace_id)
        self._validate_role(role)
        resp = await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/members",
            json={"user_id": user_id, "role": role},
        )
        return WorkspaceMember.model_validate(self._json(resp))

    async def update_member_role(
        self, workspace_id: str, user_id: str, role: str
    ) -> WorkspaceMember:
        """Change a member's role (member/admin/viewer only)."""
        workspace_id = _normalize_workspace_id(workspace_id)
        self._validate_role(role)
        resp = await self._request(
            "PUT",
            f"/api/v1/workspaces/{workspace_id}/members/{quote(user_id, safe='')}",
            json={"role": role},
        )
        return WorkspaceMember.model_validate(self._json(resp))

    async def remove_member(self, workspace_id: str, user_id: str) -> None:
        """Remove a member from the workspace (server returns 204)."""
        workspace_id = _normalize_workspace_id(workspace_id)
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
            body["allowed_context_ids"] = allowed_context_ids
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        resp = await self._request(
            "POST", f"/api/v1/workspaces/{workspace_id}/invitations", json=body
        )
        return WorkspaceInvitation.model_validate(self._json(resp))

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
        return [WorkspaceInvitation.model_validate(row) for row in self._expect_list(resp)]

    async def revoke_invitation(self, workspace_id: str, invitation_id: int) -> None:
        """Revoke a pending invitation (server returns 200 {"success": true})."""
        workspace_id = _normalize_workspace_id(workspace_id)
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/invitations/"
            f"{_require_int(invitation_id, 'invitation_id')}",
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
        workspace_id = _normalize_workspace_id(workspace_id)
        expires_days = _require_int(expires_days, "expires_days")
        if not 1 <= expires_days <= 3650:
            raise ValueError(f"expires_days must be 1-3650, got {expires_days!r}")
        resp = await self._request(
            "POST",
            f"/api/v1/workspaces/{workspace_id}/members/"
            f"{quote(user_id, safe='')}/credentials/api-keys",
            json={"name": name, "expires_days": expires_days},
        )
        payload = self._json(resp)
        try:
            return MemberAPIKey.model_validate(payload)
        except ValidationError as exc:
            # The key already exists server-side and is force-hidden — a shape
            # mismatch must not swallow the ONE chance to see the plaintext.
            plaintext = payload.get("plaintext_key") if isinstance(payload, dict) else None
            if isinstance(plaintext, str) and plaintext:
                raise KaguraError(
                    "Server returned an unexpected mint response shape, but the "
                    f"key WAS created. Save the plaintext now: {plaintext}"
                ) from exc
            raise KaguraError(
                "Server returned an unexpected mint response shape; the key may "
                "have been created without displaying its plaintext — check "
                "`kagura auth list-keys` and revoke/re-mint if present."
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
            f"/api/v1/workspaces/{workspace_id}/members/{quote(user_id, safe='')}/credentials",
        )
        return [
            MemberAPIKey.model_validate(row) for row in self._expect_wrapped_list(resp, "api_keys")
        ]

    async def revoke_member_key(self, workspace_id: str, user_id: str, key_id: int) -> None:
        """Revoke a member's API key.

        Owner-provisioned revocations are SOFT server-side (``revoked_at``
        set, row retained for forensics); success is 200 with a status
        body, and an already-revoked key surfaces as a uniform 404.
        """
        workspace_id = _normalize_workspace_id(workspace_id)
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/members/"
            f"{quote(user_id, safe='')}/credentials/api-keys/"
            f"{_require_int(key_id, 'key_id')}",
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
        *,
        request_json: dict[str, Any] | None,
        request_params: dict[str, Any] | None,
    ) -> KaguraError:
        return KaguraConnectionError(self._format_403(extract_detail(e.response)))

    def _error_429(self, e: httpx.HTTPStatusError) -> KaguraError:
        # Invite-create quota exhaustion ("Member limit reached ...") and
        # generic rate limiting both surface as 429 — keep the server
        # message, it names the cause and the fix.
        return KaguraQuotaError(
            extract_detail(e.response) or "Quota exceeded. Try again later.",
            retry_after=_retry_after_seconds(e.response),
        )
