"""Tests for WorkspaceClient (#225) — wire models, client methods, error hints.

Wire shapes assert the memory-cloud v0.42.0 contract verified against the
server source (issues #1164/#1165): canonical error envelope
``{"error", "message", "details"}``, integer invitation ids, no invitation
``status`` field, ``DELETE .../invitations/{id}`` → 200 ``{"success": true}``.
"""

import json

import httpx
import pytest

from kagura_memory.exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraNotFoundError,
    KaguraQuotaError,
)
from kagura_memory.models import WorkspaceInvitation, WorkspaceMember
from kagura_memory.workspace_client import (
    VALID_ASSIGNABLE_ROLES,
    VALID_INVITE_EXPIRES,
    WorkspaceClient,
)

WS = "11111111-2222-3333-4444-555555555555"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_workspace_member_ignores_unknown_fields():
    m = WorkspaceMember.model_validate(
        {
            "user_id": "google_123",
            "role": "owner",
            "user_name": "Fumikazu",
            "user_email": "f@example.com",
            "joined_at": "2026-06-01T00:00:00Z",
            "credentials_status": {"api_key_count": 1, "api_key_visible": False},
            "last_login_at": "2026-07-01T00:00:00Z",
            "allowed_context_ids": ["ctx-1"],
            "some_future_field": "ignored",  # unknown extra must not raise
        }
    )
    assert m.user_id == "google_123"
    assert m.role == "owner"
    assert m.user_name == "Fumikazu"
    # Audit fields are modeled (member list --json must not drop them):
    assert m.allowed_context_ids == ["ctx-1"]
    assert m.credentials_status == {"api_key_count": 1, "api_key_visible": False}
    assert m.last_login_at is not None


def test_workspace_member_minimal_shape():
    # POST/PUT responses carry only user_id/role/joined_at (v0.42.0)
    m = WorkspaceMember.model_validate(
        {"user_id": "u1", "role": "member", "joined_at": "2026-06-01T00:00:00Z"}
    )
    assert m.user_name is None and m.user_email is None


def test_workspace_invitation_matches_server_shape():
    inv = WorkspaceInvitation.model_validate(
        {
            "id": 7,  # integer PK — not a UUID string, not "invitation_id"
            "workspace_id": WS,  # extra
            "email": "new@example.com",
            "role": "member",
            "token": "tok_0123456789abcdef0123",
            "invitation_url": "https://memory.kagura-ai.com/invite/tok",
            "is_accepted": False,
            "is_expired": False,
            "created_at": "2026-07-01T00:00:00Z",
            "expires_at": "2026-07-08T00:00:00Z",
            "allowed_context_ids": ["ctx-1"],
            "invited_by": "google_123",  # extra
            "accepted_at": None,  # extra
            "accepted_by": None,  # extra
        }
    )
    assert inv.id == 7 and not inv.is_accepted


def test_workspace_invitation_programmatic_list_nulls_token():
    # v0.42.0 nulls token/invitation_url for API-key callers on LIST
    inv = WorkspaceInvitation.model_validate(
        {
            "id": 8,
            "email": "x@y.com",
            "role": "viewer",
            "token": None,
            "invitation_url": None,
            "is_accepted": False,
            "is_expired": False,
        }
    )
    assert inv.token is None and inv.invitation_url is None


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def make_client(handler) -> WorkspaceClient:
    client = WorkspaceClient(api_key="kagura_test")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer kagura_test"},
    )
    return client


def _envelope(code: str, message: str) -> dict:
    """memory-cloud canonical error envelope (v0.42.0)."""
    return {"error": code, "message": message, "details": {}}


def test_constructor_requires_credentials():
    with pytest.raises(ValueError, match="from_mcp_url"):
        WorkspaceClient()


def test_constructor_rejects_plain_http():
    with pytest.raises(Exception, match="HTTPS"):
        WorkspaceClient(api_key="k", base_url="http://memory.example.com")


def test_role_constants():
    assert "owner" not in VALID_ASSIGNABLE_ROLES
    assert set(VALID_ASSIGNABLE_ROLES) == {"member", "admin", "viewer"}
    assert VALID_INVITE_EXPIRES == (7, 30, 90, 365)


# ---------------------------------------------------------------------------
# Members
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_members():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/api/v1/workspaces/{WS}/members"
        return httpx.Response(
            200,
            json=[
                {
                    "user_id": "u1",
                    "role": "owner",
                    "user_email": "o@x.com",
                    "user_name": "Owner",
                    "joined_at": "2026-06-01T00:00:00Z",
                    "credentials_status": {"api_key_count": 2},
                    "last_login_at": None,
                    "allowed_context_ids": None,
                },
                {"user_id": "u2", "role": "member", "joined_at": "2026-06-02T00:00:00Z"},
            ],
        )

    async with make_client(handler) as c:
        members = await c.list_members(WS)
    assert [m.user_id for m in members] == ["u1", "u2"]
    assert members[0].role == "owner"


@pytest.mark.asyncio
async def test_add_member_posts_role():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert json.loads(request.content) == {"user_id": "u3", "role": "admin"}
        return httpx.Response(
            201, json={"user_id": "u3", "role": "admin", "joined_at": "2026-07-01T00:00:00Z"}
        )

    async with make_client(handler) as c:
        m = await c.add_member(WS, "u3", role="admin")
    assert m.role == "admin"


@pytest.mark.asyncio
async def test_add_member_rejects_owner_role_client_side():
    async with make_client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(ValueError, match="owner"):
            await c.add_member(WS, "u3", role="owner")


@pytest.mark.asyncio
async def test_update_member_role_encodes_user_id_segment():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.raw_path.decode()
        return httpx.Response(
            200,
            json={"user_id": "a/b#c", "role": "member", "joined_at": "2026-07-01T00:00:00Z"},
        )

    async with make_client(handler) as c:
        await c.update_member_role(WS, "a/b#c", role="member")
    assert "a%2Fb%23c" in seen["path"]  # slash and hash percent-encoded


@pytest.mark.asyncio
async def test_remove_member_returns_none_on_204():
    async with make_client(lambda r: httpx.Response(204)) as c:
        assert await c.remove_member(WS, "u2") is None


@pytest.mark.asyncio
async def test_workspace_id_validated_before_wire():
    async with make_client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(ValueError, match="UUID"):
            await c.list_members("../../etc")


# ---------------------------------------------------------------------------
# Invitations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_invitation_member_requires_contexts_client_side():
    async with make_client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(ValueError, match="allowed_context_ids"):
            await c.create_invitation(WS, "new@x.com", role="member")


@pytest.mark.asyncio
async def test_create_invitation_rejects_non_preset_expiry():
    async with make_client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(ValueError, match="7, 30, 90, 365"):
            await c.create_invitation(WS, "new@x.com", role="admin", expires_in_days=14)


@pytest.mark.asyncio
async def test_create_invitation():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/api/v1/workspaces/{WS}/invitations"
        assert json.loads(request.content) == {
            "email": "new@x.com",
            "role": "member",
            "allowed_context_ids": ["ctx-1"],
            "expires_in_days": 7,
        }
        return httpx.Response(
            201,
            json={
                "id": 7,
                "email": "new@x.com",
                "role": "member",
                "token": "tok_0123456789abcdef0123",
                "invitation_url": "https://memory.kagura-ai.com/invite/tok",
                "is_accepted": False,
                "is_expired": False,
            },
        )

    async with make_client(handler) as c:
        inv = await c.create_invitation(
            WS, "new@x.com", allowed_context_ids=["ctx-1"], expires_in_days=7
        )
    assert inv.id == 7 and inv.invitation_url is not None


@pytest.mark.asyncio
async def test_create_invitation_admin_needs_no_contexts():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"email": "adm@x.com", "role": "admin"}
        return httpx.Response(201, json={"id": 9, "email": "adm@x.com", "role": "admin"})

    async with make_client(handler) as c:
        inv = await c.create_invitation(WS, "adm@x.com", role="admin")
    assert inv.id == 9


@pytest.mark.asyncio
async def test_list_invitations_passes_include_accepted():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json=[
                {
                    "id": 8,
                    "email": "x@y.com",
                    "role": "viewer",
                    "token": None,
                    "invitation_url": None,
                    "is_accepted": True,
                    "is_expired": False,
                }
            ],
        )

    async with make_client(handler) as c:
        invs = await c.list_invitations(WS, include_accepted=True)
    assert seen["params"] == {"include_accepted": "true"}
    assert invs[0].is_accepted is True and invs[0].token is None


@pytest.mark.asyncio
async def test_revoke_invitation_returns_none_on_200_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == f"/api/v1/workspaces/{WS}/invitations/7"
        return httpx.Response(200, json={"success": True})  # NOT 204

    async with make_client(handler) as c:
        assert await c.revoke_invitation(WS, 7) is None


# ---------------------------------------------------------------------------
# Error mapping (v0.42.0 canonical envelope)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_401_maps_to_auth_error():
    async with make_client(
        lambda r: httpx.Response(401, json=_envelope("AUTH-100", "Not authenticated"))
    ) as c:
        with pytest.raises(KaguraAuthError):
            await c.list_members(WS)


@pytest.mark.asyncio
async def test_uniform_403_gets_owner_key_hint():
    async with make_client(
        lambda r: httpx.Response(403, json=_envelope("AUTH-101", "Insufficient permissions"))
    ) as c:
        with pytest.raises(KaguraConnectionError, match="OWNER's API key"):
            await c.list_members(WS)


@pytest.mark.asyncio
async def test_specific_403_passes_through_verbatim():
    # OAuth rejection / deployment kill-switch messages are already
    # actionable — the client must not rewrite them as a role failure.
    msg = (
        "Owner-API-key member management is disabled on this deployment. "
        "Use a workspace-owner session."
    )
    async with make_client(lambda r: httpx.Response(403, json=_envelope("AUTH-101", msg))) as c:
        with pytest.raises(KaguraConnectionError, match="disabled on this deployment"):
            await c.list_members(WS)


@pytest.mark.asyncio
async def test_plan_gate_403_passes_through():
    msg = "Team invitations require Pro plan. Upgrade your plan to invite team members."
    async with make_client(lambda r: httpx.Response(403, json=_envelope("HTTP-403", msg))) as c:
        with pytest.raises(KaguraConnectionError, match="Pro plan"):
            await c.create_invitation(WS, "a@b.com", role="admin")


@pytest.mark.asyncio
async def test_404_envelope_message_surfaces():
    # Uniform confinement 404 (#963): workspace-scoped key vs foreign path
    async with make_client(
        lambda r: httpx.Response(404, json=_envelope("RES-001", "Workspace not found"))
    ) as c:
        with pytest.raises(KaguraNotFoundError, match="Workspace not found"):
            await c.list_members(WS)


@pytest.mark.asyncio
async def test_404_legacy_detail_shape_surfaces():
    # Some raw-HTTPException 404s still use FastAPI's {"detail": ...}
    async with make_client(
        lambda r: httpx.Response(404, json={"detail": "Invitation 7 not found"})
    ) as c:
        with pytest.raises(KaguraNotFoundError, match="Invitation 7 not found"):
            await c.revoke_invitation(WS, 7)


@pytest.mark.asyncio
async def test_429_quota_message_kept():
    msg = "Member limit reached (5 seats). Current members: 4, Pending invitations: 1."
    async with make_client(
        lambda r: httpx.Response(
            429, json=_envelope("HTTP-429", msg), headers={"Retry-After": "30"}
        )
    ) as c:
        with pytest.raises(KaguraQuotaError, match="Member limit reached") as exc_info:
            await c.create_invitation(WS, "a@b.com", role="admin")
    assert exc_info.value.retry_after == 30


@pytest.mark.asyncio
async def test_422_validation_envelope_names_field():
    body = {
        "error": "VAL-001",
        "message": "Request validation failed",
        "details": {
            "errors": [
                {
                    "loc": ["body", "role"],
                    "msg": "Value error, role=owner invitations are not supported",
                    "type": "value_error",
                }
            ]
        },
    }
    async with make_client(lambda r: httpx.Response(422, json=body)) as c:
        with pytest.raises(KaguraConnectionError, match="body.role"):
            await c.create_invitation(WS, "a@b.com", role="admin")


# ---------------------------------------------------------------------------
# Member API keys (#201, memory-cloud#1165)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mint_member_key_posts_name_and_expiry():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/api/v1/workspaces/{WS}/members/google_2/credentials/api-keys"
        assert json.loads(request.content) == {"name": "ci-bot", "expires_days": 90}
        return httpx.Response(
            201,
            json={
                "id": 42,
                "name": "ci-bot",
                "key_prefix": "kagura_abcdef123",
                "plaintext_key": "kagura_abcdef1234567890",
                "is_visible": False,
                "visibility_expires_at": None,
                "created_at": "2026-07-03T00:00:00Z",
                "last_used_at": None,
                "revoked_at": None,
                "expires_at": "2026-10-01T00:00:00Z",
                "bound_context_id": None,
            },
        )

    async with make_client(handler) as c:
        key = await c.mint_member_key(WS, "google_2", "ci-bot", 90)
    assert key.id == 42
    assert key.plaintext_key == "kagura_abcdef1234567890"
    assert key.expires_at is not None


@pytest.mark.asyncio
async def test_mint_member_key_validates_expiry_bounds():
    async with make_client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(ValueError, match="1-3650"):
            await c.mint_member_key(WS, "google_2", "ci-bot", 0)
        with pytest.raises(ValueError, match="1-3650"):
            await c.mint_member_key(WS, "google_2", "ci-bot", 3651)


@pytest.mark.asyncio
async def test_mint_self_target_403_passes_through():
    msg = "An owner key cannot mint keys for itself. Use session self-mint."
    async with make_client(lambda r: httpx.Response(403, json=_envelope("AUTH-101", msg))) as c:
        with pytest.raises(KaguraConnectionError, match="cannot mint keys for itself"):
            await c.mint_member_key(WS, "owner-self", "x", 30)


@pytest.mark.asyncio
async def test_mint_admin_target_403_passes_through():
    msg = "Owner-provisioned keys can only be minted for member/viewer targets, not role='admin'."
    async with make_client(lambda r: httpx.Response(403, json=_envelope("AUTH-101", msg))) as c:
        with pytest.raises(KaguraConnectionError, match="member/viewer targets"):
            await c.mint_member_key(WS, "admin-user", "x", 30)


@pytest.mark.asyncio
async def test_list_member_keys_parses_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/api/v1/workspaces/{WS}/members/google_2/credentials"
        return httpx.Response(
            200,
            json={
                "api_keys": [
                    {
                        "id": 42,
                        "name": "ci-bot",
                        "key_prefix": "kagura_abcdef123",
                        "plaintext_key": None,  # always null programmatically
                        "is_visible": False,
                        "created_at": "2026-07-03T00:00:00Z",
                        "revoked_at": None,
                        "expires_at": "2026-10-01T00:00:00Z",
                    }
                ],
                "target_user_role": "member",
            },
        )

    async with make_client(handler) as c:
        keys = await c.list_member_keys(WS, "google_2")
    assert len(keys) == 1
    assert keys[0].id == 42 and keys[0].plaintext_key is None


@pytest.mark.asyncio
async def test_revoke_member_key_accepts_200_status_body():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert (
            request.url.path == f"/api/v1/workspaces/{WS}/members/google_2/credentials/api-keys/42"
        )
        return httpx.Response(200, json={"status": "revoked", "key_id": 42})

    async with make_client(handler) as c:
        assert await c.revoke_member_key(WS, "google_2", 42) is None


@pytest.mark.asyncio
async def test_member_key_user_id_segment_encoded():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.raw_path.decode()
        return httpx.Response(200, json={"api_keys": [], "target_user_role": "member"})

    async with make_client(handler) as c:
        await c.list_member_keys(WS, "a/b#c")
    assert "a%2Fb%23c" in seen["path"]


# ---------------------------------------------------------------------------
# Review-fix regressions (xhigh code review, 2026-07-03)
# ---------------------------------------------------------------------------


def test_constructor_bakes_bearer_header():
    # The MockTransport tests swap _client, so assert the REAL constructor
    # wiring here (mirrors test_files_client's header assertion).
    import asyncio

    c = WorkspaceClient(api_key="kagura_real_test")
    try:
        assert c._client.headers["Authorization"] == "Bearer kagura_real_test"
        assert c._client.headers["User-Agent"].startswith("kagura-memory-sdk/")
        assert c._oauth is None
    finally:
        asyncio.run(c.close())


def test_constructor_oauth_mode_records_handle():
    import asyncio

    def fake_auth(request):  # httpx accepts a callable auth
        return request

    c = WorkspaceClient(_oauth=fake_auth)  # type: ignore[arg-type]
    try:
        assert c._oauth is not None
        assert "Authorization" not in c._client.headers
    finally:
        asyncio.run(c.close())


@pytest.mark.asyncio
async def test_workspace_id_normalized_to_canonical():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json=[])

    async with make_client(handler) as c:
        # Windows registry format: braces + uppercase → canonical lowercase
        await c.list_members("{11111111-2222-3333-4444-555555555555}")
    assert seen["path"] == f"/api/v1/workspaces/{WS}/members"


@pytest.mark.asyncio
async def test_destructive_ids_require_strict_int():
    async with make_client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(ValueError, match="invitation_id must be an integer"):
            await c.revoke_invitation(WS, 7.9)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="key_id must be an integer"):
            await c.revoke_member_key(WS, "u2", 42.9)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="expires_days must be an integer"):
            await c.mint_member_key(WS, "u2", "x", 90.5)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_mint_salvages_plaintext_on_shape_mismatch():
    from kagura_memory.exceptions import KaguraError

    def handler(request: httpx.Request) -> httpx.Response:
        # Server drift: required fields renamed, but the one-time secret is there
        return httpx.Response(201, json={"plaintext_key": "kagura_salvaged_secret"})

    async with make_client(handler) as c:
        with pytest.raises(KaguraError, match="kagura_salvaged_secret"):
            await c.mint_member_key(WS, "u2", "ci-bot", 30)


@pytest.mark.asyncio
async def test_non_json_200_maps_to_kagura_error():
    async with make_client(lambda r: httpx.Response(200, text="<html>maintenance</html>")) as c:
        with pytest.raises(KaguraConnectionError, match="non-JSON body"):
            await c.list_members(WS)


@pytest.mark.asyncio
async def test_list_endpoint_rejects_non_list_body():
    async with make_client(lambda r: httpx.Response(200, json={"oops": True})) as c:
        with pytest.raises(KaguraConnectionError, match="expected a JSON array"):
            await c.list_invitations(WS)


@pytest.mark.asyncio
async def test_403_with_credential_marker_is_scrubbed():
    # A hostile/buggy 403 echoing credentials must not be printed verbatim.
    msg = "Denied for Authorization: Bearer kagura_leaked_key_value"
    async with make_client(lambda r: httpx.Response(403, json=_envelope("AUTH-101", msg))) as c:
        with pytest.raises(KaguraConnectionError) as exc_info:
            await c.list_members(WS)
    assert "kagura_leaked_key_value" not in str(exc_info.value)
    assert "OWNER's API key" in str(exc_info.value)  # falls back to the hint


@pytest.mark.asyncio
async def test_401_oauth_client_hints_relogin():
    def fake_auth(request):
        return request

    c = WorkspaceClient(_oauth=fake_auth)  # type: ignore[arg-type]
    await c._client.aclose()
    c._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "x"}))
    )
    async with c:
        with pytest.raises(KaguraAuthError, match="kagura auth login"):
            await c.list_members(WS)


# ---------------------------------------------------------------------------
# Coverage completion (codecov/patch on PR #228)
# ---------------------------------------------------------------------------


def test_from_mcp_url_static_source(monkeypatch):
    import asyncio

    from kagura_memory._auth import _StaticAuth

    resolved = _StaticAuth(api_key="kagura_env_x", mcp_url="https://test.com/mcp", source="env")
    monkeypatch.setattr("kagura_memory.workspace_client._resolve_auth", lambda **kw: resolved)
    c = WorkspaceClient.from_mcp_url()
    try:
        assert c.base_url == "https://test.com"
        assert c._auth_source == "env"
        assert c._oauth is None
    finally:
        asyncio.run(c.close())


def test_from_resolved_auth_oauth_branch():
    import asyncio

    from kagura_memory._auth import _OAuthAuth

    def fake_auth(request):
        return request

    resolved = _OAuthAuth(
        oauth=fake_auth,  # type: ignore[arg-type]
        mcp_url="https://test.com/mcp/w/abc",
        workspace_id=WS,
    )
    c = WorkspaceClient._from_resolved_auth(resolved)
    try:
        assert c.base_url == "https://test.com"
        assert c._auth_source == "oauth"
        assert c._workspace_id_hint == WS
        assert c._oauth is not None
    finally:
        asyncio.run(c.close())


@pytest.mark.asyncio
async def test_mint_shape_mismatch_without_plaintext_points_at_recovery():
    from kagura_memory.exceptions import KaguraError

    async with make_client(lambda r: httpx.Response(201, json={"unexpected": True})) as c:
        with pytest.raises(KaguraError, match="list-keys"):
            await c.mint_member_key(WS, "u2", "ci-bot", 30)


@pytest.mark.asyncio
async def test_list_member_keys_rejects_non_dict_body():
    async with make_client(lambda r: httpx.Response(200, json=[1, 2])) as c:
        with pytest.raises(KaguraConnectionError, match="api_keys"):
            await c.list_member_keys(WS, "u2")


@pytest.mark.asyncio
async def test_uniform_403_hint_names_static_credential_source():
    c = WorkspaceClient(api_key="kagura_test", _auth_source="config", _workspace_id_hint=WS)
    await c._client.aclose()
    c._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(403, json=_envelope("AUTH-101", "Insufficient permissions"))
        )
    )
    async with c:
        with pytest.raises(KaguraConnectionError) as exc_info:
            await c.list_members(WS)
    msg = str(exc_info.value)
    assert "credential source" in msg
    assert WS[:8] in msg  # workspace hint prefix, never the full wire value


@pytest.mark.asyncio
async def test_generic_status_maps_to_connection_error():
    async with make_client(lambda r: httpx.Response(500, json=_envelope("SYS-001", "boom"))) as c:
        with pytest.raises(KaguraConnectionError, match="HTTP 500: boom"):
            await c.list_members(WS)


@pytest.mark.asyncio
async def test_network_error_maps_to_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope", request=request)

    async with make_client(handler) as c:
        with pytest.raises(KaguraConnectionError, match="Connection failed"):
            await c.list_members(WS)


@pytest.mark.asyncio
async def test_list_member_keys_rejects_null_or_non_list_api_keys_field():
    # Copilot review (PR #228): guard the api_keys FIELD, not just the envelope.
    async with make_client(
        lambda r: httpx.Response(200, json={"api_keys": None, "target_user_role": "member"})
    ) as c:
        with pytest.raises(KaguraConnectionError, match="api_keys"):
            await c.list_member_keys(WS, "u2")
    async with make_client(lambda r: httpx.Response(200, json={"api_keys": {"oops": 1}})) as c:
        with pytest.raises(KaguraConnectionError, match="api_keys"):
            await c.list_member_keys(WS, "u2")
