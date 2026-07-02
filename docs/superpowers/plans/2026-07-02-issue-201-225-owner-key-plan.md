# Issues #201 + #225 (owner-key only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship workspace member management (#225) and API-key provisioning (#201) as **owner-API-key-only** operational tooling — CLI + `WorkspaceClient` in this SDK, unblocked by two scoped memory-cloud server changes.

**Architecture:** memory-cloud's member/invitation/credential endpoints are session-cookie-only today; two server PRs make them accept programmatic auth **gated at workspace-owner role**. The SDK then adds a `WorkspaceClient` (mirroring `FilesClient`'s shape) plus `kagura workspace member|invite` and `kagura auth create-key` CLI commands.

**Tech Stack:** Python 3.11+, httpx (async), click, pydantic v2 (`extra="ignore"` wire models), pytest + httpx.MockTransport + click CliRunner, uv/ruff/pyright. Server side: FastAPI (memory-cloud).

## Global Constraints

- **Owner-key only (user directive 2026-07-02):** every new programmatic surface in this plan requires the **workspace owner's** credential. Web-UI (session) semantics on the server are unchanged.
- SDK repo rules: branch `{issue}-{type}/{desc}` from `main`, Conventional Commits, `/quality` → `/simplify` → `/self-review` before PR, squash merge (`.claude/rules/development-workflow.md`).
- Wire models are non-strict: `model_config = ConfigDict(extra="ignore")` (lesson from #222/#226).
- Never persist a minted API key to disk; print once + stderr warning (mirrors `kagura auth token` / `resource tokens create`).
- Never store API keys as instance attributes (existing `python.md` rule; `FilesClient` pattern).
- `user_id` / `invitation_id` path segments must be URL-encoded (`urllib.parse.quote(..., safe="")`).
- No MCP tool exposure for member mutation (explicit #225 decision — privileged owner ops stay off the agent tool surface).
- Verify request/response schemas against the live server OpenAPI (`GET /openapi.json`, tags `workspaces` + `invitations`) before wiring each client method — #226 taught that guessed shapes break.

---

## Part 0 — Verified server reality (2026-07-02, memory-cloud `origin/main` @ `fe6d7376`)

Facts checked against source; line numbers are origin/main of that commit.

| Surface | File | Auth today | Role gate today |
|---|---|---|---|
| `GET /api/v1/workspaces/{id}/members` | `backend/src/api/routes/workspaces.py:574` | **session-cookie only** (`get_current_user` ← `request.state.user` set only by cookie `SessionMiddleware`) | member+ |
| `POST .../members` (add) | `workspaces.py:692` | session only | admin+ |
| `PUT .../members/{user_id}` (set-role) | `workspaces.py:724` | session only | admin+; #254 guards: no self-role-change, only owner may change owner |
| `DELETE .../members/{user_id}` | `workspaces.py:842` | session only | **owner only** (#217) |
| Invitations create/list/revoke | `backend/src/api/routes/invitations.py:47,173,241` | session only | admin gated via `check_workspace_access` |
| `GET .../member-quota` | `invitations.py:501` | session only | — |
| API-key self-mint | `backend/src/api/routes/api_keys.py:191` | `SessionUser` **by explicit decision #252** ("Session-only authentication (no API keys)") | — |
| Member-credential mint / delete | `backend/src/api/routes/member_credentials.py:277,530` | `SessionUser` | **strictly self-only** — mint 403s unless path `user_id` == session user; `check_can_manage` excludes `create` for owner/admin (verified 2026-07-02 adversarial review) |

Enablers that already exist server-side (no need to invent):

- `APIKeyOrSessionUser` (`auth/dependencies.py:460`) — dual/triple-mode: **OAuth Bearer with RFC 6750 scopes (landed 2026-05-15, PR #652)** → API key → session. Used by `files.py` (all endpoints).
- `require_workspace_owner` (`dependencies.py:572`, #276) and `require_workspace_admin` (`dependencies.py:680`, #398) are **already dual-mode** ("Accepts both session auth and API key auth (for SDK/CI automation)"), with WARN audit logs on deny.
- `require_workspace_admin_session` (`dependencies.py:628`, #398) is the precedent for *auth-mode-differentiated* gating (session-only for billing).

**Consequences:**

1. **#225 cannot be implemented SDK-first.** An API-key call to any members endpoint 401s today. Server change required (Part A1).
2. **#201's original blocker stands** for OAuth self-mint (`api_keys.py` is SessionUser by design #252), **but** the old note "production /api/v1/* does not validate OAuth Bearer" is outdated — #652 fixed that for `APIKeyOrSessionUser` routes. The remaining gap is only the mint endpoints' deliberate session-only gate.

---

## Part 0.5 — Decisions

### D1 (settled by user directive): #225 auth posture = owner-key only

- Server: members + invitations endpoints accept **API-key principals only, gated at OWNER role** on the *path* workspace — stricter than the session path (which keeps list=member+, add/set-role=admin+, remove=owner). **OAuth Bearer → 403 on these endpoints** (2026-07-02 adversarial review: every `kagura auth login` device token carries `memory:read memory:write`, so plain `APIKeyOrSessionUser` would turn every MCP token into a member-management credential — the opposite of "owner キーのみ"). Rationale: conservative first release of a privileged surface; trivially relaxable later; `require_workspace_admin_session` is the precedent for differentiating by auth mode.
- SDK: normal credential resolver chain (`env > OAuth profile > .kagura.json`) — no special-casing; the server enforces. CLI maps 403 → "requires workspace **owner**" hint.

### D2 (recommendation — confirm before Phase C): #201 under "owner-key only"

**Recommended: (a) owner-key provisioning via member-credentials.** Do NOT touch `api_keys.py` (#252 stays intact: a key can never mint a key *for its own principal escalation*). Instead make `member_credentials.py` mint/delete dual-mode gated by `require_workspace_owner`. Then `kagura auth create-key --user <member-id>` lets an owner provision workspace keys for member/service accounts from CI or a headless box — which is exactly the sharp use case in #201's motivation. Risk is bounded: owner-key → member-scoped key is privilege *down*grade, and denials/mints get audit logs.
- Issue #201 is then **closed** with a decision note: OAuth *self*-mint stays deferred (dashboard workaround documented in the issue body); owner-key provisioning covers headless/CI.

**Alternative (b): OAuth self-mint endpoint** with a dedicated elevated scope (`apikey:create`). Bigger security design (short-lived token minting long-lived credential = escalation; needs fresh-auth/consent design). Defer unless the user specifically wants self-mint.

---

## Part A — memory-cloud server work (separate repo, own sessions/plans)

These are **issue drafts to file on kagura-ai/memory-cloud**. Each gets its own in-repo plan per that repo's conventions; this SDK plan only pins the contract the SDK will consume.

### A1 — feat(workspaces): owner-key programmatic access to member + invitation endpoints

```markdown
## Summary
Member/invitation management is session-cookie-only (`get_current_user`), so the
Python SDK / CLI cannot automate it (kagura-memory-python-sdk#225). Open the 8
endpoints below to programmatic auth — API keys only, gated at workspace-OWNER
role on the PATH workspace — keeping session semantics identical.

## Endpoints (8; session gates unchanged)
- workspaces.py: GET(member+)/POST(admin+) /api/v1/workspaces/{id}/members,
  PUT(admin+ #254)/DELETE(owner #217) .../members/{user_id}
- invitations.py: POST/GET(admin+) .../invitations,
  DELETE(admin+) .../invitations/{invitation_id:int}, GET(member+) .../member-quota
- NOT changed: POST /invitations/accept, GET /invitations/pending,
  public GET /invitations/{token}

## Contract
- Session principal: behavior unchanged.
- API-key principal: `PermissionService.check_workspace_owner(user_id,
  path_workspace_id)` IN-HANDLER for every endpoint (require_workspace_owner reads
  only current_workspace_id, never the path param — wrong gate here).
- OAuth Bearer principal: **403 on every endpoint** — device tokens carry
  memory:read/write and must not become member-management credentials; revisit
  only with a dedicated workspace:admin scope.
- Principal discrimination (NEW pattern; shared helper, fail-closed): api-key ⇔
  `"api_key_workspace_id" in user` (key-PRESENCE, never .get() truthiness — global
  keys carry the key with value None); oauth ⇔ `"oauth_scope" in user`; session ⇔
  `"sub" in user`.
- Workspace-scoped key vs foreign path workspace → **uniform 404** (#963 pattern:
  pass key_workspace_id into PermissionService as contexts.py:751 / memory.py do;
  files.py has NO such confinement — known gap, not a precedent).
- Programmatic add/set-role/invite accept roles member|admin|viewer;
  **role=owner → 422** (no owner-level persistence from a leaked key).
- Programmatic invitations LIST omits `token`/`invitation_url` (bearer
  join-credentials must not land in CI logs); token only in POST create response.
- Rate limit: add ENDPOINT_RATE_LIMITS entries (10/min) for POST invitations +
  member mutations (today: generic tier limit, up to 1000/min on Pro).
- Audit: log programmatic SUCCESSES (add/set-role/remove/invite create+revoke)
  with actor + key prefix + target; WARN on denies (#389 pattern).
- Pinned wire shapes for SDK: WorkspaceInvitationResponse has **id:int** (no
  `status` field — derive from is_accepted/is_expired), token, invitation_url,
  expires_at|null, allowed_context_ids|null; DELETE invitations returns **200
  {"success": true}** (not 204); invite create requires email (must match
  invitee's Google account), role member|viewer requires allowed_context_ids
  (min 1) / ≥1 shared context; 403 "Team invitations require Pro plan" on
  free/basic (detail strings must stay distinguishable from the role 403); 429 on
  member-quota exceeded.

## Tests
- API-key owner (global + matching scoped): all 8 endpoints succeed.
- API-key admin/member/viewer: 403 everywhere. OAuth Bearer (any scope): 403.
- Session admin/member: no regression. Scoped key vs foreign workspace: uniform 404.
- Global key owner-of-A vs path B (not owner of B): denied.
- Programmatic role=owner: 422. Programmatic invite list: token absent.
- Success + deny paths write audit entries.
```

### A2 — feat(credentials): owner-key minting/revocation of member API keys *(only if D2 = (a))*

```markdown
## Summary
`POST .../credentials/api-keys` (member_credentials.py:276) and the per-id DELETE
(:530) are SessionUser-gated **and strictly self-only** (owner/admin cannot act on
others even in the browser — check_can_manage excludes `create`). This issue adds a
NEW capability: owner-provisioned mint/list/revoke of OTHER members' keys via the
owner's API key (kagura-memory-python-sdk#201). Session self-mint/self-delete
semantics unchanged; `api_keys.py` stays session-only — #252 is NOT relaxed.

## Contract
- Auth framework identical to A1: APIKeyOrSessionUser; OAuth Bearer → 403;
  in-handler check_workspace_owner(user_id, path_workspace_id); scoped-key
  mismatch → uniform 404; shared fail-closed principal discriminator.
- Anti key-self-replication (#252 threat made explicit): programmatic mint 403s
  when target user_id == caller user_id, and 403s unless the target member's role
  is member|viewer — strictly privilege-DOWNGRADE provisioning. (Minted keys carry
  no role of their own: they act as the target user with their live workspace
  role; platform role hardcoded "user".)
- Expiry: CreateAPIKeyRequest has NO expiry field today → keys never expire. Add
  `expires_days: int | None` (1–3650, mirroring /api/v1/config/api-keys), plumb to
  APIKeyManager.create_key; **required for owner-provisioned mints** (400 if
  omitted).
- Plaintext hygiene: today plaintext is re-revealed via GET while is_visible
  (auto_hide default 10min, encrypted at rest since Migration 035). Programmatic
  mint forces hidden_at=now (rejects client auto_hide_minutes) → plaintext_key
  ONLY in the 201 response; programmatic GET list is always metadata-only; and
  `bound_context_id` is rejected (400) — public-bound keys (#626) stay self-only.
- List endpoint EXISTS — extend, don't add: GET .../members/{user_id}/credentials
  (:77) already gives session owner/admin metadata; accept the owner API-key
  principal. Envelope: MemberCredentialsResponse {api_keys:
  list[MemberAPIKeyResponse], target_user_role} — NOT a bare array.
  MemberAPIKeyResponse: **id:int**, name, key_prefix, **plaintext_key** (field
  name, not api_key), is_visible, visibility_expires_at, created_at,
  last_used_at, revoked_at, bound_context_id.
- Revocation: per-id DELETE is a HARD delete today. Relax self-only →
  self-or-workspace-owner (same target-role restriction); owner-provisioned
  revocation is a SOFT revoke (revoked_at=now, row retained for forensics);
  audit entry written before the state change; pin the success status code in
  OpenAPI at impl time.
- Rate limit: add ENDPOINT_RATE_LIMITS entry (10/min) for member-credentials
  paths (currently generic tier limits only). Audit: AuditLog for
  owner-provisioned mint/revoke (new — regular key creation is unaudited today;
  only public-bound keys write AuditLog).

## Tests
- Owner key mints for member/viewer target: 201 with plaintext_key; follow-up GET:
  plaintext_key null. Mint for SELF: 403. Target owner/admin: 403.
  bound_context_id: 400. expires_days omitted: 400.
- member/admin/viewer key: 403. OAuth Bearer: 403. Foreign workspace: uniform 404.
- Session self-mint/self-delete: unchanged. Owner revoke: soft (revoked_at set,
  row retained); audit entries for mint+revoke; plaintext never in logs.
```

**Gate:** SDK Phase B merges only after A1 is deployed to production (blue-green); Phase C after A2. Local development against `docker-compose` memory-cloud running the A1/A2 branch is fine before that.

---

## Part B — SDK #225: `WorkspaceClient` + `kagura workspace` CLI

Branch: `225-feat/workspace-member-cli`. All tasks in this repo.

### Task B1: Wire models — `WorkspaceMember`, `WorkspaceInvitation`

**Files:**
- Modify: `src/kagura_memory/models.py` (append after `FileObject` block)
- Test: `tests/test_workspace_client.py` (new)

**Interfaces:**
- Produces: `WorkspaceMember(user_id: str, role: str, user_name: str | None, user_email: str | None, joined_at: datetime | None)`; `WorkspaceInvitation(id: int, email: str | None, role: str, token: str | None, invitation_url: str | None, is_accepted: bool, is_expired: bool, created_at: datetime | None, expires_at: datetime | None, allowed_context_ids: list[str] | None)`. Both `extra="ignore"`. **Server shape verified 2026-07-02:** invitation `id` is an integer PK and there is NO `status` field — pending is derived from `is_accepted`/`is_expired`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_workspace_client.py
"""Tests for WorkspaceClient (#225) — wire models, client methods, error hints."""

from kagura_memory.models import WorkspaceInvitation, WorkspaceMember


def test_workspace_member_ignores_unknown_fields():
    m = WorkspaceMember.model_validate(
        {
            "user_id": "google_123",
            "role": "owner",
            "user_name": "Fumikazu",
            "user_email": "f@example.com",
            "joined_at": "2026-06-01T00:00:00Z",
            "credentials_status": "active",  # server extra — must not raise
        }
    )
    assert m.user_id == "google_123"
    assert m.role == "owner"
    assert m.user_name == "Fumikazu"


def test_workspace_member_minimal_shape():
    # POST/PUT responses carry only user_id/role/joined_at (workspaces.py)
    m = WorkspaceMember.model_validate(
        {"user_id": "u1", "role": "member", "joined_at": "2026-06-01T00:00:00Z"}
    )
    assert m.user_name is None and m.user_email is None


def test_workspace_invitation_matches_server_shape():
    inv = WorkspaceInvitation.model_validate(
        {
            "id": 7,  # integer PK — not a UUID string, not "invitation_id"
            "workspace_id": "11111111-2222-3333-4444-555555555555",  # extra
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
        }
    )
    assert inv.id == 7 and not inv.is_accepted


def test_workspace_invitation_programmatic_list_omits_token():
    # A1 contract: server omits token/invitation_url on programmatic LIST
    inv = WorkspaceInvitation.model_validate(
        {"id": 8, "email": "x@y.com", "role": "viewer",
         "is_accepted": False, "is_expired": False}
    )
    assert inv.token is None and inv.invitation_url is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_workspace_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'WorkspaceMember'`

- [ ] **Step 3: Implement models**

```python
# src/kagura_memory/models.py (append; follow FileObject's commenting style)


class WorkspaceMember(BaseModel):
    """A workspace member row (#225).

    ``extra="ignore"`` — the server adds display fields over time
    (e.g. ``credentials_status``); deserialization must not break (#222 lesson).
    ``user_name``/``user_email`` are populated by the list endpoint only;
    add/set-role responses carry the minimal shape.
    """

    model_config = ConfigDict(extra="ignore")

    user_id: str
    role: str
    user_name: str | None = None
    user_email: str | None = None
    joined_at: datetime | None = None


class WorkspaceInvitation(BaseModel):
    """A workspace invitation (#225). Non-strict like WorkspaceMember.

    Server shape (``WorkspaceInvitationResponse``): ``id`` is an INTEGER PK
    and there is no ``status`` field — pending is derived from
    ``is_accepted``/``is_expired``. ``token``/``invitation_url`` are bearer
    join-credentials: the server omits them on programmatic LIST responses,
    so they are optional here and the CLI prints them only on create.
    """

    model_config = ConfigDict(extra="ignore")

    id: int
    email: str | None = None
    role: str
    token: str | None = None
    invitation_url: str | None = None
    is_accepted: bool = False
    is_expired: bool = False
    created_at: datetime | None = None
    expires_at: datetime | None = None
    allowed_context_ids: list[str] | None = None
```

- [ ] **Step 4: Run tests — PASS**, then `uv run ruff check src/ tests/ && uv run pyright src/`

- [ ] **Step 5: Commit**

```bash
git add src/kagura_memory/models.py tests/test_workspace_client.py
git commit -m "feat(workspace): add WorkspaceMember/WorkspaceInvitation wire models (#225)"
```

### Task B2: `WorkspaceClient` — member methods

**Files:**
- Create: `src/kagura_memory/workspace_client.py`
- Modify: `src/kagura_memory/__init__.py` (export `WorkspaceClient`)
- Test: `tests/test_workspace_client.py`

**Interfaces:**
- Consumes: `WorkspaceMember` (B1); `_resolve_auth`/`KaguraOAuth`/`_AuthSource` from `kagura_memory.auth` + `_auth` (same imports as `files_client.py`); `raise_for_kagura_status` from `_http`.
- Produces:
  - `WorkspaceClient.__init__(api_key=None, base_url="https://memory.kagura-ai.com", timeout=30.0, *, _oauth=None, _auth_source=None, _workspace_id_hint=None)` — same contract as `FilesClient.__init__` (ValueError without api_key/_oauth; HTTPS validation; key never stored as attribute).
  - `WorkspaceClient.from_mcp_url(...)` / `WorkspaceClient._from_resolved_auth(auth, *, workspace_id_hint=None)` — copy `FilesClient`'s factories verbatim (adjust class name).
  - `async list_members(workspace_id: str) -> list[WorkspaceMember]` → `GET /api/v1/workspaces/{workspace_id}/members`
  - `async add_member(workspace_id: str, user_id: str, role: str = "member") -> WorkspaceMember` → `POST .../members` body `{"user_id":..., "role":...}`
  - `async update_member_role(workspace_id: str, user_id: str, role: str) -> WorkspaceMember` → `PUT .../members/{user_id}` body `{"role":...}`
  - `async remove_member(workspace_id: str, user_id: str) -> None` → `DELETE .../members/{user_id}` (expect 204)
  - `VALID_ASSIGNABLE_ROLES = ("member", "admin", "viewer")` module constant — `owner` deliberately excluded (A1 server contract 422s `role=owner` from programmatic principals; owner promotion belongs to future transfer-ownership). `viewer` confirmed assignable (server regex `^(owner|admin|member|viewer)$`, `workspaces.py:289-299`).
  - async context manager (`__aenter__`/`__aexit__`/`close`) like `FilesClient`.

- [ ] **Step 1: Failing tests** (httpx.MockTransport, same pattern as `tests/test_files_client.py`)

```python
# tests/test_workspace_client.py (append)
import json

import httpx
import pytest

from kagura_memory.workspace_client import WorkspaceClient

WS = "11111111-2222-3333-4444-555555555555"


def make_client(handler) -> WorkspaceClient:
    client = WorkspaceClient(api_key="kagura_test")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer kagura_test"},
    )
    return client


@pytest.mark.asyncio
async def test_list_members():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"/api/v1/workspaces/{WS}/members"
        return httpx.Response(
            200,
            json=[
                {"user_id": "u1", "role": "owner", "user_email": "o@x.com",
                 "joined_at": "2026-06-01T00:00:00Z", "credentials_status": "active"},
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
        return httpx.Response(200, json={"user_id": "a/b#c", "role": "member",
                                         "joined_at": "2026-07-01T00:00:00Z"})

    async with make_client(handler) as c:
        await c.update_member_role(WS, "a/b#c", role="member")
    assert "a%2Fb%23c" in seen["path"]  # slash and hash percent-encoded


@pytest.mark.asyncio
async def test_remove_member_returns_none_on_204():
    async with make_client(lambda r: httpx.Response(204)) as c:
        assert await c.remove_member(WS, "u2") is None
```

- [ ] **Step 2: Run — FAIL** (`ModuleNotFoundError: kagura_memory.workspace_client`)

- [ ] **Step 3: Implement**

```python
# src/kagura_memory/workspace_client.py
"""REST client for workspace member & invitation management (#225).

Owner-key operational tooling: every endpoint requires the WORKSPACE
OWNER's credential when called programmatically (server contract,
memory-cloud A1). Mirrors :class:`~kagura_memory.files_client.FilesClient`
for construction/auth; see that module for the resolver rationale.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from ._http import raise_for_kagura_status, validate_https_url
from .models import WorkspaceInvitation, WorkspaceMember

VALID_ASSIGNABLE_ROLES = ("member", "admin", "viewer")
# "owner" is excluded: promotion overlaps the (out-of-scope) transfer-ownership
# flow, and the server 422s role=owner from programmatic principals (A1).


class WorkspaceClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://memory.kagura-ai.com",
        timeout: float = 30.0,
        *,
        _oauth: "KaguraOAuth | None" = None,
        _auth_source: "_AuthSource | None" = None,
        _workspace_id_hint: str | None = None,
    ) -> None:
        # ... copy FilesClient.__init__ body: ValueError guard, HTTPS
        # validation, httpx.AsyncClient with either _oauth auth= or the
        # baked Bearer header. Key never stored as an attribute.
        ...

    # from_mcp_url / _from_resolved_auth / __aenter__ / __aexit__ / close:
    # copy from FilesClient verbatim (class name adjusted).

    def _validate_role(self, role: str) -> None:
        if role not in VALID_ASSIGNABLE_ROLES:
            raise ValueError(
                f"role must be one of {VALID_ASSIGNABLE_ROLES}, got {role!r}. "
                "Promoting to 'owner' is not supported from the CLI."
            )

    async def list_members(self, workspace_id: str) -> list[WorkspaceMember]:
        resp = await self._request("GET", f"/api/v1/workspaces/{workspace_id}/members")
        return [WorkspaceMember.model_validate(row) for row in resp.json()]

    async def add_member(
        self, workspace_id: str, user_id: str, role: str = "member"
    ) -> WorkspaceMember:
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
        self._validate_role(role)
        resp = await self._request(
            "PUT",
            f"/api/v1/workspaces/{workspace_id}/members/{quote(user_id, safe='')}",
            json={"role": role},
        )
        return WorkspaceMember.model_validate(resp.json())

    async def remove_member(self, workspace_id: str, user_id: str) -> None:
        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{workspace_id}/members/{quote(user_id, safe='')}",
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            resp = await self._client.request(method, f"{self.base_url}{path}", **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            raise_for_kagura_status(e)  # NoReturn — maps 401/429/…; B3 adds 403/404 hints
```

(実装時は `...` を FilesClient の該当実装で埋める — plan 上の省略は「同一コードの重複回避」であり、コピー元が明示されているため。)

- [ ] **Step 4: Export + run tests — PASS**; `ruff` + `pyright` clean

```python
# src/kagura_memory/__init__.py — add alongside FilesClient export
from .workspace_client import WorkspaceClient  # noqa: F401  (+ __all__ entry)
```

- [ ] **Step 5: Commit** — `feat(workspace): WorkspaceClient member list/add/set-role/remove (#225)`

### Task B3: 403/404 mapping with owner-key hints

**Files:**
- Modify: `src/kagura_memory/workspace_client.py` (`_request`)
- Test: `tests/test_workspace_client.py`

**Interfaces:**
- Produces: 403 → `KaguraAuthDeniedError` that **preserves the server detail** and appends the owner hint only for the uniform authz shape (`"Insufficient permissions"`): `"workspace owner API key required — member management is owner-only when called programmatically; OAuth tokens are not accepted (workspace=<id>)"`. Invitation create can 403 for a *plan gate* (`"Team invitations require Pro plan"`) — that detail must pass through untouched, not be rewritten as a role failure. 404 on `add_member` → `KaguraNotFoundError("user not found — use `kagura workspace invite create <email>` to invite a new user")`; 404 elsewhere → plain `KaguraNotFoundError` with server detail. Note: cross-workspace use of a scoped key surfaces as a **uniform 404** (#963), not 403.

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.asyncio
async def test_403_maps_to_owner_hint():
    from kagura_memory.exceptions import KaguraAuthDeniedError

    async with make_client(
        lambda r: httpx.Response(403, json={"detail": "Insufficient permissions"})
    ) as c:
        with pytest.raises(KaguraAuthDeniedError, match="owner"):
            await c.list_members(WS)


@pytest.mark.asyncio
async def test_add_member_404_hints_invite():
    from kagura_memory.exceptions import KaguraNotFoundError

    async with make_client(
        lambda r: httpx.Response(404, json={"detail": "User not found"})
    ) as c:
        with pytest.raises(KaguraNotFoundError, match="kagura workspace invite create"):
            await c.add_member(WS, "ghost-user")
```

- [ ] **Step 2: FAIL** → **Step 3: implement** (branch on `e.response.status_code` inside `_request` before delegating to `raise_for_kagura_status`; `add_member` catches `KaguraNotFoundError` and re-raises with the invite hint) → **Step 4: PASS + lint** → **Step 5: Commit** `feat(workspace): actionable 403/404 hints for owner-key member ops (#225)`

### Task B4: Invitation methods

**Files:**
- Modify: `src/kagura_memory/workspace_client.py`
- Test: `tests/test_workspace_client.py`

**Interfaces:**
- Produces: `create_invitation(workspace_id, email, role="member", *, allowed_context_ids: list[str] | None = None, expires_in_days: int | None = None) -> WorkspaceInvitation` — client-side `ValueError` when `role in ("member", "viewer")` and no `allowed_context_ids` (server requires ≥1, `invitation_service.py:129-146`); `list_invitations(workspace_id, include_accepted: bool = False) -> list[WorkspaceInvitation]`; `revoke_invitation(workspace_id, invitation_id: int) -> None` — server returns **200 `{"success": true}`**, not 204, and `invitation_id` is an `int` (no URL-encoding concern).
- Paths **verified 2026-07-02 against origin/main** (`invitations.py:47,173,241` + `main.py:589`): `POST/GET /api/v1/workspaces/{workspace_id}/invitations`, `DELETE /api/v1/workspaces/{workspace_id}/invitations/{invitation_id}`.
- [ ] **Step 1: Failing tests** — mirror B2's style: create asserts the full body (`email`/`role`/`allowed_context_ids`/`expires_in_days`) and 201→`WorkspaceInvitation`; list returns parsed array; revoke expects DELETE + 200 `{"success": true}`→None; `create_invitation` validates role via `_validate_role` and the member/viewer `allowed_context_ids` precondition.

```python
@pytest.mark.asyncio
async def test_create_invitation_member_requires_contexts_client_side():
    async with make_client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(ValueError, match="allowed_context_ids"):
            await c.create_invitation(WS, "new@x.com", role="member")


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
        return httpx.Response(201, json={
            "id": 7, "email": "new@x.com", "role": "member",
            "token": "tok_0123456789abcdef0123",
            "invitation_url": "https://memory.kagura-ai.com/invite/tok",
            "is_accepted": False, "is_expired": False,
        })

    async with make_client(handler) as c:
        inv = await c.create_invitation(
            WS, "new@x.com", allowed_context_ids=["ctx-1"], expires_in_days=7
        )
    assert inv.id == 7 and inv.invitation_url is not None


@pytest.mark.asyncio
async def test_revoke_invitation_returns_none_on_200_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == f"/api/v1/workspaces/{WS}/invitations/7"
        return httpx.Response(200, json={"success": True})  # NOT 204

    async with make_client(handler) as c:
        assert await c.revoke_invitation(WS, 7) is None
```

- [ ] **Steps 2–5:** FAIL → implement (same `_request` plumbing; `invitation_id` is an int path segment; `create_invitation` validates role via `_validate_role` and raises `ValueError` for member/viewer without `allowed_context_ids`) → PASS + lint → commit `feat(workspace): invitation create/list/revoke (#225)`

### Task B5: CLI — `kagura workspace member` + `kagura workspace invite`

**Files:**
- Modify: `src/kagura_memory/cli.py`
- Test: `tests/test_cli_workspace.py` (new; mirror `tests/test_cli_files.py`: CliRunner + `patch("kagura_memory.cli.WorkspaceClient")`)

**Interfaces:**
- Consumes: `WorkspaceClient` (B2–B4); `_resolve_auth`, `_resolve_workspace_from_source`, `_bound_workspace_for_hint`, `_exc_message` (existing cli.py helpers, `cli.py:1779` region).
- Produces CLI surface:

```
kagura workspace member list   [--workspace <id>] [--json]
kagura workspace member add    <user-id> --role member|admin|viewer [--workspace <id>]
kagura workspace member set-role <user-id> --role member|admin|viewer [--workspace <id>]
kagura workspace member remove <user-id> [--yes] [--workspace <id>]
kagura workspace invite create <email> --role member|admin|viewer
                               [--context <uuid>]... [--expires-days N] [--workspace <id>]
kagura workspace invite list   [--workspace <id>] [--json]
kagura workspace invite revoke <invitation-id> [--workspace <id>]
```

Token hygiene: `invite create` prints `invitation_url` (and token) **once**; `invite list` renders metadata only — the server omits `token`/`invitation_url` for programmatic principals (A1), and the CLI must not try to resurface them. `--context` (repeatable) maps to `allowed_context_ids`, required for `--role member|viewer`.

Notes: `invite create` is a subcommand (not `kagura workspace invite <email>` as sketched in the issue) — click groups can't cleanly take positionals, and it mirrors the existing `kagura resource tokens create/list/revoke` family. Default workspace = same-source resolution via `_resolve_workspace_from_source` (issue #115 rule: never mix one source's key with another's workspace); `--workspace` overrides. `remove` prompts `click.confirm` unless `--yes`.

- [ ] **Step 1: Failing tests** (representative — full file covers each command):

```python
# tests/test_cli_workspace.py
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from kagura_memory.cli import main
from kagura_memory.models import WorkspaceMember

WS = "11111111-2222-3333-4444-555555555555"


def _mock_client(mock_cls, **method_returns):
    inst = MagicMock()
    inst.__aenter__ = AsyncMock(return_value=inst)
    inst.__aexit__ = AsyncMock(return_value=False)
    for name, value in method_returns.items():
        setattr(inst, name, AsyncMock(return_value=value))
    mock_cls._from_resolved_auth.return_value = inst
    return inst


@patch("kagura_memory.cli.WorkspaceClient")
def test_member_list_renders_table(mock_cls, mock_config):
    _mock_client(mock_cls, list_members=[
        WorkspaceMember(user_id="u1", role="owner", user_email="o@x.com"),
    ])
    result = CliRunner().invoke(
        main, ["workspace", "member", "list", "--workspace", WS]
    )
    assert result.exit_code == 0
    assert "owner" in result.output and "u1" in result.output


@patch("kagura_memory.cli.WorkspaceClient")
def test_member_remove_requires_confirmation(mock_cls, mock_config):
    inst = _mock_client(mock_cls, remove_member=None)
    result = CliRunner().invoke(
        main, ["workspace", "member", "remove", "u2", "--workspace", WS], input="n\n"
    )
    assert result.exit_code != 0 or "Aborted" in result.output
    inst.remove_member.assert_not_called()


@patch("kagura_memory.cli.WorkspaceClient")
def test_member_remove_yes_skips_prompt(mock_cls, mock_config):
    inst = _mock_client(mock_cls, remove_member=None)
    result = CliRunner().invoke(
        main, ["workspace", "member", "remove", "u2", "--yes", "--workspace", WS]
    )
    assert result.exit_code == 0
    inst.remove_member.assert_awaited_once_with(WS, "u2")
```

(`mock_config` fixture: reuse/extract the one in `tests/test_cli_files.py` — it patches `load_config` + `_resolve_auth` to a static-key auth; move it to `tests/conftest.py` if not already shared.)

- [ ] **Step 2: FAIL** → **Step 3: implement.** Add `_run_workspace_command(operation, workspace_id)` — a thin sibling of `_run_files_command` (`cli.py:1866`) that always resolves workspace (`needs_context=True` equivalent), constructs `WorkspaceClient._from_resolved_auth(auth, workspace_id_hint=_bound_workspace_for_hint(auth, config))`, and passes the resolved/overridden workspace UUID to the operation. Then the click groups:

```python
@main.group()
def workspace():
    """Workspace administration (owner API key required)."""


@workspace.group()
def member():
    """Manage workspace members (owner-only, server v0.42.0+)."""


@member.command("list")
@click.option("--workspace", "workspace_id", help="Workspace UUID (default: bound workspace)")
@click.option("--json", "as_json", is_flag=True, default=False)
def member_list(workspace_id: str | None, as_json: bool) -> None:
    """List members with role / email / joined date."""

    async def _op(client: WorkspaceClient, ws: str) -> str:
        members = await client.list_members(ws)
        if as_json:
            return json.dumps([m.model_dump(mode="json") for m in members], indent=2)
        lines = [f"{'USER':<28} {'ROLE':<8} {'EMAIL':<30} JOINED"]
        for m in members:
            joined = m.joined_at.date().isoformat() if m.joined_at else "-"
            lines.append(f"{m.user_id:<28} {m.role:<8} {m.user_email or '-':<30} {joined}")
        return "\n".join(lines)

    _run_workspace_command(_op, workspace_id)
```

`add`/`set-role`: `@click.option("--role", type=click.Choice(["member", "admin", "viewer"]), required=True)`. `remove`: `if not yes: click.confirm(f"Remove {user_id} from workspace {ws}?", abort=True)`. `invite` group mirrors `member` (create/list/revoke; create adds `--context` multiple + `--expires-days`). Every command's docstring notes the owner-API-key requirement (OAuth tokens rejected server-side).

- [ ] **Step 4: PASS + lint/typecheck** → **Step 5: Commit** `feat(cli): kagura workspace member/invite command groups (#225)`

### Task B6: Docs + integration smoke

**Files:**
- Modify: `README.md` (CLI reference + short "manage members from the CLI" example), `CLAUDE.md` (Key SDK Features + CLI list), `skills/` guide if it enumerates CLI commands
- Modify: `tests/test_workspace_integration.py` (new; opt-in like `tests/test_files_integration.py` — skipped unless env creds present)

- [ ] **Step 1:** README section (example: list → invite → set-role → remove with `--yes`), explicitly: "These commands require the **workspace owner's** API key; admin/member keys receive 403."
- [ ] **Step 2:** Integration test gated on `KAGURA_INTEGRATION=1` + owner key env: `list_members` round-trip only (non-destructive).
- [ ] **Step 3:** `uv run pytest && uv run ruff check src/ tests/ && uv run pyright src/` — all green.
- [ ] **Step 4: Commit** `docs(workspace): member management CLI reference + integration smoke (#225)`

### Task B7: Gate + PR

- [ ] Run `/quality` → `/simplify` → `/self-review`; fix all `[C]`/`[W]`.
- [ ] Confirm server A1 is deployed to production (or PR notes the version gate "server v0.42.0+").
- [ ] PR: `feat(workspace): member management client + kagura workspace CLI (#225)` — body links #225, notes owner-key-only posture (D1) and the A1 server dependency. Squash-merge after approval; then `/release minor`.

---

## Part C — SDK #201: `kagura auth create-key` (gated on D2 + A2)

Branch: `201-feat/auth-create-key`. **Do not start until D2 is confirmed and A2 is merged server-side** — the issue's own decision note ("do not implement on speculation") still governs.

### Task C1: `WorkspaceClient.mint_member_key` / `list_member_keys` / `revoke_member_key`

**Files:**
- Modify: `src/kagura_memory/workspace_client.py`, `src/kagura_memory/models.py`
- Test: `tests/test_workspace_client.py`

**Interfaces (pinned by A2's contract; re-verify against merged OpenAPI):**
- `MemberAPIKey(id: int, name: str, key_prefix: str, plaintext_key: str | None, is_visible: bool, visibility_expires_at: datetime | None, created_at: datetime | None, last_used_at: datetime | None, revoked_at: datetime | None)` — `extra="ignore"`; **`id` is `int` and the plaintext field is `plaintext_key`** (not `api_key`), non-null only in the mint 201 response (print-once).
- `async mint_member_key(workspace_id, user_id, name, expires_days: int) -> MemberAPIKey` → `POST /api/v1/workspaces/{id}/members/{user_id}/credentials/api-keys` — `expires_days` **required** (server 400s without it, per A2); never send `bound_context_id`; server 403s self-targets and owner/admin targets (member/viewer service identities only).
- `async list_member_keys(workspace_id, user_id) -> list[MemberAPIKey]` → `GET .../members/{user_id}/credentials` — parse the `MemberCredentialsResponse` **envelope** (`api_keys` field + `target_user_role`; NOT a bare array); always metadata-only (`plaintext_key` null).
- `async revoke_member_key(workspace_id, user_id, key_id: int) -> None` → `DELETE .../credentials/api-keys/{key_id}` (soft revoke server-side per A2; pin the success status code from merged OpenAPI).

- [ ] TDD cycle identical in structure to B2/B4 (failing MockTransport tests asserting method/path/body → implement → pass → commit `feat(workspace): owner-key member API-key mint/list/revoke (#201)`).

### Task C2: CLI — `kagura auth create-key` family

**Files:**
- Modify: `src/kagura_memory/cli.py`
- Test: `tests/test_cli_auth_keys.py` (new)

**Surface (issue #201's proposed names, `--user` marks the owner-provisioning semantics):**

```
kagura auth create-key --user <member-user-id> --name <n> --expires-days N [--workspace <id>]
kagura auth list-keys  --user <member-user-id> [--workspace <id>] [--json]
kagura auth revoke-key <key-id> --user <member-user-id> [--workspace <id>]
```

`--expires-days` is **required** (A2: the server 400s owner-provisioned mints without it — never-expiring CI keys are not allowed). `--user` must be another member with role member|viewer; self-targets 403 server-side (anti key-self-replication).

- [ ] `create-key` prints the key **once to stdout**, warning to **stderr**: `⚠ Save this key now — it cannot be shown again.` (mirror `resource tokens create`, `cli.py:1239` region). Never written to config/credentials files.
- [ ] `revoke-key` prompts `click.confirm` unless `--yes`.
- [ ] 403 hint: "workspace owner API key required". Tests: mint prints key + warning; key absent from any file writes; revoke confirms; TDD as in B5.
- [ ] Commit `feat(cli): kagura auth create-key/list-keys/revoke-key via owner key (#201)`

### Task C3: Close the loop on issue #201

- [ ] Update issue #201 with the D2(a) decision comment: OAuth *self*-mint remains deferred (workaround = dashboard, documented in issue body); owner-key provisioning shipped for the headless/CI case. Close via the PR (`Closes #201`).
- [ ] Docs (README auth section) + `/quality` → `/simplify` → `/self-review` → PR → `/release minor`.

---

## Sequencing & gates

```
D1 ✔ (user directive)          D2 confirm (recommend (a))
        │                              │
        ▼                              ▼
memory-cloud A1  ──deploy──►  SDK Phase B (#225)  ──merge──► release
memory-cloud A2  ──deploy──►  SDK Phase C (#201)  ──merge──► release
```

- A1 and A2 are independent server PRs; B and C are independent SDK PRs. B does not block C or vice versa; both block on their server halves.
- Local development may proceed against a docker-compose memory-cloud running the A1/A2 branches before production deploy, but SDK PRs merge only after the server contract is deployed (posture from #226: never ship SDK ahead of the server shape).

## Risks / open items

1. ~~Invitation & member-credential wire shapes unverified~~ — **verified 2026-07-02** by a 5-agent adversarial review workflow (`wf_a710fae0-0ed`): invitation/key `id` fields are `int`, no invitation `status` field, `DELETE .../invitations/{id}` returns 200 `{"success": true}`, member-key list is an envelope, mint request today is `{name, auto_hide_minutes, bound_context_id}` with **no expiry field**. Re-verify against OpenAPI once A1/A2 merge (fields may evolve during server implementation).
2. ~~`viewer` role unconfirmed~~ — **confirmed assignable** on all three surfaces (`^(owner|admin|member|viewer)$`); included in `VALID_ASSIGNABLE_ROLES` and CLI choices.
3. **member-quota endpoint** deliberately out of v1 (not in #225 acceptance criteria); revisit if invite UX needs a capacity pre-check.
4. **`set-role owner`** stays excluded (server #254 guards + A1's programmatic 422 + future transfer-ownership command).
5. **D2(b)** (OAuth self-mint) remains available if the user rejects (a); it re-opens the #252 security design and would replace Part C's server half with a scope/consent design issue on memory-cloud.
6. **OAuth-profile users**: all Part B/C commands are API-key-only server-side (OAuth Bearer → 403 by A1/A2 contract). The CLI resolver chain may pick an OAuth profile; the B3 hint covers this ("OAuth tokens are not accepted").
7. **403 disambiguation**: invitation create 403s for the Pro-plan gate as well as the role gate — B3 passes server details through instead of blanket-rewriting 403s.
8. **Pre-existing server hole found during review**: a session ADMIN can create a `role=owner` invitation (`invitations.py:47` is admin-gated and `WorkspaceInvitationCreate.role` accepts the full enum), bypassing the #254 owner-change guard on the role-update path — file as its own memory-cloud bug alongside A1/A2.
