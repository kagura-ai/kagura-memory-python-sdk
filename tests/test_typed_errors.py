"""Typed server errors from memory-cloud v0.78.0-v0.141.0 (#299, typed-errors section).

The wire shapes follow memory-cloud v0.141.0:

- MCP refusals and server failures share one envelope (``mcp_server/tools/
  _errors.py``): ``{status, error, message, help, ...}``, plus ``cause`` /
  ``correlation_id`` / ``retryable`` / ``outcome`` on a server failure.
- The Free capacity lock (``services/capacity_lock.py``,
  ``utils/exceptions.CapacityLockedError``): REST ``403 CAPACITY-001`` with
  ``details.gate == "capacity"``, MCP ``capacity_locked``.
- ``/mcp`` scope refusal (``mcp_server/transport._reject_insufficient_scope``):
  HTTP 403, ``WWW-Authenticate: Bearer error="insufficient_scope"`` and a
  JSON-RPC error whose ``data`` names ``required_scope``; REST
  (``auth/dependencies.py``) sends the challenge with ``scope="<required>"``.
- 503: ``/mcp``'s ``temporarily_unavailable`` and JSON-RPC transport failures
  (``transport._send_auth_unavailable`` / ``_send_transport_failure``), REST
  ``DB-002`` with ``details.reason`` (``api/main._database_unavailable_response``).
"""

from __future__ import annotations

import copy
import pickle
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from click.testing import CliRunner

import kagura_memory
from kagura_memory import (
    KaguraCapacityLockedError,
    KaguraServiceUnavailableError,
    UsageInfo,
)
from kagura_memory._http import ErrorBody, gate_error, raise_for_kagura_status
from kagura_memory.agents_client import AgentsClient
from kagura_memory.cli import _cli_error_message, main
from kagura_memory.client import KaguraClient
from kagura_memory.exceptions import (
    KaguraAuthError,
    KaguraConnectionError,
    KaguraError,
    KaguraFeatureNotAvailableError,
    KaguraNotFoundError,
    KaguraPartialRollbackError,
    KaguraQuotaError,
)
from kagura_memory.files_client import FilesClient
from kagura_memory.models import ResourceEventRequest
from kagura_memory.resource_client import ResourceClient
from kagura_memory.secrets.client import SecretClient
from kagura_memory.workspace_client import WorkspaceClient
from tests.conftest import kagura_client_on

WS = "11111111-2222-3333-4444-555555555555"
CTX = "22222222-3333-4444-5555-666666666666"
CLEANUP_URL = "https://memory.kagura-ai.com/workspace/settings/plan"

# CapacityLock.as_dict() of a workspace 12 memories and 0 bytes over Free.
_LOCK_FIGURES = {
    "memory_count": 1012,
    "memory_limit": 1000,
    "over_memories": 12,
    "used_bytes": 1024,
    "storage_limit_bytes": 104857600,
    "over_bytes": 0,
    "cleanup_url": CLEANUP_URL,
}
_LOCK_MESSAGE = (
    "This workspace is over its Free plan capacity (12 memories over). Search and saving "
    "are paused until it is cleaned up or re-subscribed."
)
_LOCK_HELP = f"Remove 12 memories or re-subscribe at {CLEANUP_URL}. Until then ..."


# ---------------------------------------------------------------------------
# KaguraError: code, details and the envelope properties
# ---------------------------------------------------------------------------


def test_kagura_error_defaults_carry_no_server_fields():
    err = KaguraError("plain")
    assert (err.code, err.details) == (None, {})
    assert (err.help, err.cause, err.correlation_id, err.retryable, err.outcome) == (
        None,
        None,
        None,
        None,
        None,
    )


def test_kagura_error_properties_read_details_and_ignore_wrong_types():
    err = KaguraError("x")
    err.details = {
        "help": "Check with recall before calling it again.",
        "cause": "timeout",
        "correlation_id": "4bf92f3577b34da6",
        "retryable": False,
        "outcome": "unknown",
    }
    assert err.help == "Check with recall before calling it again."
    assert (err.cause, err.correlation_id) == ("timeout", "4bf92f3577b34da6")
    assert (err.retryable, err.outcome) == (False, "unknown")
    err.details = {"help": 1, "retryable": "yes", "outcome": None}
    assert (err.help, err.retryable, err.outcome) == (None, None, None)


def test_properties_tolerate_a_subclass_that_skips_init():
    class BareError(KaguraError):
        def __init__(self) -> None:  # deliberately no super().__init__
            pass

    err = BareError()
    assert (err.help, err.retryable) == (None, None)


def _round_trips(err: KaguraError) -> list[KaguraError]:
    # Unpickles only bytes this test just produced from its own object.
    return [pickle.loads(pickle.dumps(err)), copy.copy(err)]


def test_new_errors_pickle_with_their_attributes():
    plain = KaguraError("x")
    plain.code, plain.details = "payload_too_large", {"limit": 1, "actual": 2}
    lock = KaguraCapacityLockedError("locked", cleanup_url=CLEANUP_URL, over_memories=12)
    lock.code = "CAPACITY-001"
    busy = KaguraServiceUnavailableError("HTTP 503", retry_after=5, reason="pool_exhausted")
    for clone in _round_trips(plain):
        assert (clone.code, clone.details) == ("payload_too_large", {"limit": 1, "actual": 2})
    for clone in _round_trips(lock):
        assert isinstance(clone, KaguraCapacityLockedError)
        assert (clone.cleanup_url, clone.over_memories, clone.code) == (
            CLEANUP_URL,
            12,
            "CAPACITY-001",
        )
    for clone in _round_trips(busy):
        assert isinstance(clone, KaguraServiceUnavailableError)
        assert (clone.retry_after, clone.reason) == (5, "pool_exhausted")


def test_new_error_classes_place_in_the_hierarchy():
    assert issubclass(KaguraCapacityLockedError, KaguraError)
    assert not issubclass(KaguraCapacityLockedError, KaguraQuotaError)
    assert not issubclass(KaguraCapacityLockedError, KaguraFeatureNotAvailableError)
    assert issubclass(KaguraServiceUnavailableError, KaguraConnectionError)
    assert {"KaguraCapacityLockedError", "KaguraServiceUnavailableError", "CapacityLock"} <= set(
        kagura_memory.__all__
    )


# ---------------------------------------------------------------------------
# MCP envelopes → KaguraClient._raise_for_mcp_error
# ---------------------------------------------------------------------------


def _mcp_error(result: dict[str, Any], operation: str = "recall") -> KaguraError:
    with pytest.raises(KaguraError) as exc:
        KaguraClient._raise_for_mcp_error({"status": "error", **result}, operation)
    return exc.value


def test_mcp_capacity_locked_is_typed_with_the_overage():
    err = _mcp_error(
        {
            "error": "capacity_locked",
            "message": _LOCK_MESSAGE,
            "gate": "capacity",
            **_LOCK_FIGURES,
            "help": _LOCK_HELP,
        }
    )
    assert isinstance(err, KaguraCapacityLockedError)
    assert str(err).startswith("recall failed (capacity_locked): This workspace is over")
    assert (err.gate, err.code, err.cleanup_url) == ("capacity", "capacity_locked", CLEANUP_URL)
    assert (err.memory_count, err.memory_limit, err.over_memories) == (1012, 1000, 12)
    assert (err.used_bytes, err.storage_limit_bytes, err.over_bytes) == (1024, 104857600, 0)
    assert err.help == _LOCK_HELP
    assert err.details == {"gate": "capacity", **_LOCK_FIGURES, "help": _LOCK_HELP}


def test_mcp_capacity_locked_for_an_outsider_has_no_figures():
    """A public-context reader of a locked workspace: every number is omitted."""
    err = _mcp_error(
        {
            "error": "capacity_locked",
            "message": "The workspace that owns this context is over its plan's capacity.",
            "gate": "capacity",
            "help": "This context belongs to a workspace you do not manage; ask its owner.",
        }
    )
    assert isinstance(err, KaguraCapacityLockedError)
    assert (err.cleanup_url, err.memory_count, err.over_bytes) == (None, None, None)


def test_capacity_code_without_a_gate_is_still_the_lock():
    """An envelope missing ``gate`` (or a drifted one) is typed by the code."""
    err = gate_error("capacity_locked", "locked", {"over_memories": "12"})
    assert isinstance(err, KaguraCapacityLockedError)
    assert err.over_memories is None  # malformed figure dropped, refusal kept


def test_mcp_payload_too_large_stays_a_plain_error_with_its_numbers():
    """v0.108.0 (#2003): the 1 MB guard is no quota — split the memory instead."""
    err = _mcp_error(
        {
            "error": "payload_too_large",
            "message": "Memory size 1,200,000 characters exceeds limit 1,000,000 characters.",
            "limit": 1000000,
            "actual": 1200000,
            "help": "One memory holds up to 1,000,000 characters ...",
        },
        "remember",
    )
    assert type(err) is KaguraError
    assert err.code == "payload_too_large"
    assert (err.details["limit"], err.details["actual"]) == (1000000, 1200000)
    assert err.help == "One memory holds up to 1,000,000 characters ..."


def test_mcp_version_conflict_keeps_its_code():
    """v0.113.0 (#2043): typed later by #302; until then the code identifies it."""
    err = _mcp_error(
        {
            "error": "version_conflict",
            "message": "Memory version is 4, not 3.",
            "help": "Read the target's current state before deciding whether to call again.",
        },
        "update_memory",
    )
    assert type(err) is KaguraError
    assert err.code == "version_conflict"


def test_mcp_server_failure_keeps_the_whole_envelope():
    """v0.82.0 (#1742): a write whose outcome is unknown is not retryable."""
    help_text = (
        "This tool changes data, and the change may or may not have been applied. "
        "Check with recall before calling it again, so it is not applied twice."
    )
    err = _mcp_error(
        {
            "error": "service_unavailable",
            "message": "remember could not reach a service it depends on.",
            "cause": "service_unavailable",
            "help": help_text,
            "correlation_id": "0af7651916cd43dd8448eb211c80319c",
            "retryable": False,
            "outcome": "unknown",
        },
        "remember",
    )
    assert type(err) is KaguraError
    assert err.code == "service_unavailable"
    assert (err.cause, err.retryable, err.outcome) == ("service_unavailable", False, "unknown")
    assert err.correlation_id == "0af7651916cd43dd8448eb211c80319c"
    assert err.help == help_text
    assert not {"status", "error", "message"} & err.details.keys()


def test_mcp_not_found_and_partial_rollback_keep_the_envelope_too():
    not_found = _mcp_error(
        {"error": "context_not_found", "message": "Context not found", "help": "Check the id."}
    )
    assert isinstance(not_found, KaguraNotFoundError)
    assert (not_found.code, not_found.help) == ("context_not_found", "Check the id.")
    rollback = _mcp_error(
        {
            "error": "partial_rollback",
            "message": "Rollback completed with 1 error(s).",
            "report_id": "rid-1",
        },
        "rollback_sleep_run",
    )
    assert isinstance(rollback, KaguraPartialRollbackError)
    assert (rollback.code, rollback.details["report_id"]) == ("partial_rollback", "rid-1")


def test_typed_gate_errors_get_the_code_as_well():
    err = _mcp_error(
        {"error": "feature_not_available", "message": "nope", "gate": "plan", "feature": "x"}
    )
    assert isinstance(err, KaguraFeatureNotAvailableError)
    assert err.code == "feature_not_available"


@pytest.mark.parametrize("quota_type", ["content_chunk_contexts", "llm_selector_contexts"])
def test_new_tier_quota_types_are_quotas_without_a_gate(quota_type):
    """v0.113.0 (#2103) / v0.115.0 (#2144) caps on a server that sends no ``gate``."""
    err = gate_error("QUOTA-001", "cap", {"quota_type": quota_type, "limit": 1, "current": 1})
    assert isinstance(err, KaguraQuotaError)
    assert (err.quota_type, err.code) == (quota_type, "QUOTA-001")


# ---------------------------------------------------------------------------
# get_usage → UsageInfo.capacity_lock
# ---------------------------------------------------------------------------

_USAGE = {
    "status": "success",
    "plan": "free",
    "memories": {"used": 1012, "limit": 1000, "percentage": 101.2},
    "contexts": {"used": 2, "limit": 3},
    "members": {"used": 1, "limit": 1},
    "mcp_calls_per_day": {"used": 10, "limit": 100},
}


async def _get_usage(result: dict[str, Any]) -> UsageInfo:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    try:
        with patch.object(client, "_call_tool", new_callable=AsyncMock) as mock:
            mock.return_value = result
            return await client.get_usage()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_usage_reads_the_capacity_lock():
    usage = await _get_usage({**_USAGE, "capacity_lock": {**_LOCK_FIGURES, "help": _LOCK_HELP}})
    lock = usage.capacity_lock
    assert lock is not None
    assert (lock.over_memories, lock.over_bytes, lock.cleanup_url) == (12, 0, CLEANUP_URL)
    assert (lock.help, lock.purge_at) == (_LOCK_HELP, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{"capacity_lock": None}, {}], ids=["null", "older-server"])
async def test_get_usage_without_a_lock(extra):
    assert (await _get_usage({**_USAGE, **extra})).capacity_lock is None


def test_capacity_lock_purge_at_is_nullable():
    usage = UsageInfo.model_validate(
        {**_USAGE, "capacity_lock": {**_LOCK_FIGURES, "purge_at": "2026-12-01T04:00:00Z"}}
    )
    assert usage.capacity_lock is not None
    assert usage.capacity_lock.purge_at == datetime(2026, 12, 1, 4, tzinfo=UTC)


# ---------------------------------------------------------------------------
# REST envelopes → the REST clients
# ---------------------------------------------------------------------------


def _rest(cls, handler):
    client = cls(api_key="kagura_test", base_url="https://test.com")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer kagura_test"},
    )
    return client


def _respond(status: int, body: Any, **headers: str):
    return lambda _req: httpx.Response(status, json=body, headers=headers)


_CAPACITY_BODY = {
    "error": "CAPACITY-001",
    "message": _LOCK_MESSAGE,
    "details": {"gate": "capacity", **_LOCK_FIGURES},
}


_EVENT = ResourceEventRequest(op="upsert", doc_id="sku-1", payload={"name": "x"})


async def _files_upload(c: FilesClient) -> Any:
    return await c.upload(context_id=CTX, source=b"x", filename="a.txt")


async def _agents_bootstrap(c: AgentsClient) -> Any:
    return await c.bootstrap(WS, context_id=CTX)


async def _resource_ingest(c: ResourceClient) -> Any:
    return await c.ingest_event("products", "rk_test", _EVENT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls", "call"),
    [
        (FilesClient, _files_upload),
        (AgentsClient, _agents_bootstrap),
        (ResourceClient, _resource_ingest),
    ],
    ids=["files-upload", "agents-bootstrap", "resource-ingest"],
)
async def test_rest_capacity_001_is_typed_on_every_rest_client(cls, call):
    """It used to be ``KaguraConnectionError("HTTP 403 ...")``."""
    async with _rest(cls, _respond(403, _CAPACITY_BODY)) as c:
        with pytest.raises(KaguraCapacityLockedError, match="over its Free plan") as exc:
            await call(c)
    err = exc.value
    assert (err.code, err.gate, err.cleanup_url, err.over_memories) == (
        "CAPACITY-001",
        "capacity",
        CLEANUP_URL,
        12,
    )
    assert err.details == _CAPACITY_BODY["details"]


def _seat_body(feature: str) -> dict[str, Any]:
    """``plan_seats.paused_details``: FEAT-001 with the seats or slots."""
    return {
        "error": "FEAT-001",
        "message": f"Your plan covers fewer {feature} than this workspace has.",
        "details": {
            "feature": feature,
            "gate": "plan",
            "limit": 1,
            "current": 3,
            "required_plan": "pro",
            "required_plan_display": "L",
            "current_plan": "free",
        },
    }


@pytest.mark.asyncio
async def test_rest_seatless_member_on_a_workspace_route_is_a_feature_gate():
    """v0.100.0 (#1943): ``feature: members`` on every workspace route."""
    async with _rest(WorkspaceClient, _respond(403, _seat_body("members"))) as c:
        with pytest.raises(KaguraFeatureNotAvailableError) as exc:
            await c.list_members(WS)
    assert (exc.value.feature, exc.value.code) == ("members", "FEAT-001")
    assert (exc.value.details["limit"], exc.value.details["current"]) == (1, 3)


@pytest.mark.asyncio
async def test_rest_ingest_without_the_plan_allowance_is_a_feature_gate():
    """v0.99.0 (#1939): resource-token ingest answers FEAT-001 when the allowance is 0."""
    body = {
        "error": "FEAT-001",
        "message": "Feature 'resources' not available on Free plan.",
        "details": {"feature": "resources", "gate": "plan", "current_plan": "free"},
    }
    async with _rest(ResourceClient, _respond(403, body)) as c:
        with pytest.raises(KaguraFeatureNotAvailableError) as exc:
            await c.ingest_events("products", "rk_test", [_EVENT])
    assert (exc.value.feature, exc.value.gate) == ("resources", "plan")


# ---------------------------------------------------------------------------
# 403 insufficient_scope (memory-cloud v0.79.0+, #1686)
# ---------------------------------------------------------------------------

_MCP_SCOPE_CHALLENGE = (
    'Bearer realm="Kagura Memory Cloud", error="insufficient_scope", '
    'error_description="This tool needs the memory:write scope", '
    'scope="memory:read memory:write", '
    'resource_metadata="https://memory.kagura-ai.com/.well-known/oauth-protected-resource"'
)
_MCP_SCOPE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "error": {
        "code": -32002,
        "message": "remember needs the OAuth scope memory:write, which this connection's "
        "authorization does not grant.",
        "data": {
            "error": "insufficient_scope",
            "required_scope": "memory:write",
            "help": "Reconnect this server in your MCP client and approve access that "
            "includes memory:write, then call the tool again.",
        },
    },
}


def _mcp_client(handler) -> KaguraClient:
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._session_id = "pre-set-session"
    return client


@pytest.mark.asyncio
async def test_mcp_insufficient_scope_is_an_auth_error_naming_the_scope():
    """It used to be ``KaguraConnectionError("HTTP 403: ...")``."""
    handler = _respond(403, _MCP_SCOPE_BODY, **{"WWW-Authenticate": _MCP_SCOPE_CHALLENGE})
    client = _mcp_client(handler)
    try:
        with pytest.raises(KaguraAuthError, match="lacks the memory:write scope") as exc:
            await client.remember(context_id=CTX, summary="summary text", content="c")
    finally:
        await client.close()
    err = exc.value
    assert err.code == "insufficient_scope"
    # data.required_scope wins over the challenge's granted+required scope.
    assert err.details["required_scope"] == "memory:write"
    assert "kagura auth login" in str(err)
    assert err.help is not None and err.help.startswith("Reconnect this server")


@pytest.mark.asyncio
async def test_rest_insufficient_scope_challenge_is_an_auth_error():
    """REST sends only the challenge (``HTTP-403`` body) with the exact scope."""
    body = {"error": "HTTP-403", "message": "Insufficient scope: 'memory:write' required"}
    challenge = 'Bearer error="insufficient_scope", scope="memory:write"'
    handler = _respond(403, {**body, "details": {}}, **{"WWW-Authenticate": challenge})
    # FilesClient's own 403 hook (workspace hints) must not swallow it.
    async with _rest(FilesClient, handler) as c:
        with pytest.raises(KaguraAuthError, match="lacks the memory:write scope") as exc:
            await c.list(context_id=CTX)
    assert exc.value.code == "insufficient_scope"
    assert exc.value.details == {"required_scope": "memory:write"}


def test_insufficient_scope_without_a_scope_still_says_what_to_do():
    request = httpx.Request("POST", "https://test.com/mcp")
    response = httpx.Response(
        403,
        text="forbidden",
        headers={"WWW-Authenticate": "Bearer error=insufficient_scope"},
        request=request,
    )
    with pytest.raises(KaguraAuthError, match="lacks a scope this call needs") as exc:
        raise_for_kagura_status(httpx.HTTPStatusError("403", request=request, response=response))
    assert exc.value.details == {}


@pytest.mark.parametrize(
    "headers",
    [{}, {"WWW-Authenticate": 'Bearer error="invalid_token"'}, {"WWW-Authenticate": "Basic"}],
    ids=["no-challenge", "other-error", "other-scheme"],
)
def test_other_403s_stay_connection_errors(headers):
    request = httpx.Request("POST", "https://test.com/mcp")
    body = {"error": "access_denied", "error_description": "You are not a member."}
    response = httpx.Response(403, json=body, headers=headers, request=request)
    with pytest.raises(KaguraConnectionError, match="You are not a member") as exc:
        raise_for_kagura_status(httpx.HTTPStatusError("403", request=request, response=response))
    assert not isinstance(exc.value, KaguraAuthError)
    assert (exc.value.code, exc.value.details) == (
        "access_denied",
        {"error_description": "You are not a member."},
    )


# ---------------------------------------------------------------------------
# HTTP 503 → KaguraServiceUnavailableError
# ---------------------------------------------------------------------------


def _raised(status: int, body: Any = None, *, text: str = "", **headers: str) -> KaguraError:
    request = httpx.Request("POST", "https://test.com/mcp")
    if body is None:
        response = httpx.Response(status, text=text, headers=headers, request=request)
    else:
        response = httpx.Response(status, json=body, headers=headers, request=request)
    with pytest.raises(KaguraError) as exc:
        raise_for_kagura_status(
            httpx.HTTPStatusError(str(status), request=request, response=response)
        )
    return exc.value


def test_mcp_auth_unavailable_503():
    """v0.82.0: the credential could not be checked — not a 401."""
    err = _raised(
        503,
        {
            "error": "temporarily_unavailable",
            "error_description": "Could not verify credentials right now; retry shortly.",
            "correlation_id": "c0ffee",
        },
        **{"Retry-After": "5"},
    )
    assert isinstance(err, KaguraServiceUnavailableError)
    assert isinstance(err, KaguraConnectionError)
    assert (err.retry_after, err.reason, err.code) == (
        5,
        "temporarily_unavailable",
        "temporarily_unavailable",
    )
    assert err.correlation_id == "c0ffee"
    assert "Could not verify credentials" in str(err)


def test_mcp_transport_failure_503_reads_the_jsonrpc_data():
    data = {
        "error": "service_unavailable",
        "cause": "service_unavailable",
        "help": "No tool ran and nothing was changed, so calling it again is safe.",
        "correlation_id": "abc123",
        "retryable": True,
        "retry_after_seconds": 5,
    }
    body = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "down", "data": data}}
    err = _raised(503, body)  # no Retry-After header: data.retry_after_seconds is used
    assert isinstance(err, KaguraServiceUnavailableError)
    assert (err.retry_after, err.reason, err.retryable) == (5, "service_unavailable", True)
    assert err.help == data["help"]


def test_mcp_503_for_a_write_has_no_retry_window():
    data = {"error": "timeout", "cause": "timeout", "retryable": False, "outcome": "unknown"}
    body = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "slow", "data": data}}
    err = _raised(503, body)
    assert isinstance(err, KaguraServiceUnavailableError)
    assert (err.retry_after, err.reason, err.outcome) == (None, "timeout", "unknown")


def test_503_without_a_body_is_still_typed():
    err = _raised(503, text="Service Unavailable")
    assert isinstance(err, KaguraServiceUnavailableError)
    assert (err.retry_after, err.reason, err.code) == (None, None, None)


def test_500_keeps_the_correlation_id_on_a_connection_error():
    data = {"error": "internal_error", "cause": "internal_error", "correlation_id": "feed"}
    body = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "boom", "data": data}}
    err = _raised(500, body)
    assert type(err) is KaguraConnectionError
    assert (err.code, err.correlation_id, err.cause) == ("internal_error", "feed", "internal_error")


@pytest.mark.asyncio
async def test_rest_db_002_is_service_unavailable_with_its_reason():
    body = {
        "error": "DB-002",
        "message": "Database temporarily unavailable",
        "details": {"reason": "pool_exhausted"},
    }
    async with _rest(ResourceClient, _respond(503, body, **{"Retry-After": "5"})) as c:
        with pytest.raises(KaguraServiceUnavailableError, match="HTTP 503: Database") as exc:
            await c.list_tokens()
    assert (exc.value.retry_after, exc.value.reason, exc.value.code) == (
        5,
        "pool_exhausted",
        "DB-002",
    )


@pytest.mark.asyncio
async def test_rest_503_without_a_body_is_bare():
    async with _rest(ResourceClient, lambda _req: httpx.Response(503)) as c:
        with pytest.raises(KaguraServiceUnavailableError) as exc:
            await c.list_tokens()
    assert str(exc.value) == "HTTP 503"


# ---------------------------------------------------------------------------
# CLI error text
# ---------------------------------------------------------------------------


def test_cli_error_shows_the_cleanup_page():
    err = KaguraCapacityLockedError("locked", cleanup_url=CLEANUP_URL)
    assert _cli_error_message(err) == f"locked\n  Clean up or re-subscribe: {CLEANUP_URL}"
    assert _cli_error_message(KaguraCapacityLockedError("locked")) == "locked"


def test_cli_error_names_the_cleanup_page_once_when_help_has_it():
    """MCP's ``help`` already names the page (CapacityLockedError.help_text): no second line."""
    err = KaguraCapacityLockedError(
        "locked", cleanup_url=CLEANUP_URL, details={"gate": "capacity", "help": _LOCK_HELP}
    )
    assert _cli_error_message(err) == f"locked\n  Help: {_LOCK_HELP}"
    assert _cli_error_message(err).count(CLEANUP_URL) == 1
    # A help that does not name the page keeps the cleanup line.
    other = KaguraCapacityLockedError(
        "locked", cleanup_url=CLEANUP_URL, details={"help": "Ask the workspace owner."}
    )
    assert _cli_error_message(other) == (
        f"locked\n  Clean up or re-subscribe: {CLEANUP_URL}\n  Help: Ask the workspace owner."
    )


def test_cli_error_shows_the_retry_window():
    err = KaguraServiceUnavailableError("HTTP 503: busy", retry_after=5, reason="pool_exhausted")
    assert _cli_error_message(err) == "HTTP 503: busy\n  Retry after: 5 s"
    assert _cli_error_message(KaguraServiceUnavailableError("HTTP 503")) == "HTTP 503"


def test_cli_error_shows_help_and_correlation_id():
    err = KaguraError("remember failed (timeout): remember did not finish.")
    err.details = {"help": "Check with recall first.", "correlation_id": "abc123"}
    assert _cli_error_message(err) == (
        "remember failed (timeout): remember did not finish.\n"
        "  Help: Check with recall first.\n"
        "  Correlation ID: abc123"
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_cli_command_prints_the_new_lines(mock_client_cls, mock_config):
    """End to end through ``_run_client_command``, the path most commands take."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_client = AsyncMock()
    mock_client.recall.side_effect = KaguraCapacityLockedError(
        "recall failed (capacity_locked): locked", cleanup_url=CLEANUP_URL
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    result = CliRunner().invoke(main, ["recall", "query", "--context-id", CTX])
    assert result.exit_code == 1
    assert "recall failed (capacity_locked): locked" in result.output
    assert f"Clean up or re-subscribe: {CLEANUP_URL}" in result.output


# ---------------------------------------------------------------------------
# Review follow-ups: code/details on every REST error, challenge position,
# list_tags' remapped REST errors
# ---------------------------------------------------------------------------

_DENY_BODY = {"error": "AUTH-101", "message": "Insufficient permissions", "details": {}}


async def _files_list(c: FilesClient) -> Any:
    return await c.list(context_id=CTX)


async def _workspace_members(c: WorkspaceClient) -> Any:
    return await c.list_members(WS)


async def _secret_pubkeys(c: SecretClient) -> Any:
    return await c.list_pubkeys()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls", "call", "wording"),
    [
        (FilesClient, _files_list, "HTTP 403: Insufficient permissions"),
        (WorkspaceClient, _workspace_members, "workspace OWNER"),
        (SecretClient, _secret_pubkeys, "you may not have a grant"),
    ],
    ids=["files", "workspace", "secret"],
)
async def test_rest_client_403_hooks_keep_their_wording_and_the_code(cls, call, wording):
    """A plain 403 still goes through each client's own hook, and keeps the code."""
    async with _rest(cls, _respond(403, _DENY_BODY)) as c:
        with pytest.raises(KaguraConnectionError, match=wording) as exc:
            await call(c)
    assert not isinstance(exc.value, KaguraAuthError)
    assert (exc.value.code, exc.value.details) == ("AUTH-101", {})


@pytest.mark.asyncio
async def test_rest_409_and_404_keep_code_details_and_correlation_id():
    conflict = {
        "error": "RES-002",
        "message": "Memory version is 4, not 3.",
        "details": {"reason": "version_conflict", "correlation_id": "c0ffee"},
    }
    async with _rest(ResourceClient, _respond(409, conflict)) as c:
        with pytest.raises(KaguraConnectionError, match="HTTP 409") as exc:
            await c.list_tokens()
    assert exc.value.code == "RES-002"
    assert exc.value.details == {"reason": "version_conflict", "correlation_id": "c0ffee"}
    assert exc.value.correlation_id == "c0ffee"

    missing = {"error": "NOT-001", "message": "Agent not found", "details": {"agent_id": "a1"}}
    async with _rest(AgentsClient, _respond(404, missing)) as c:
        with pytest.raises(KaguraNotFoundError, match="Agent not found") as exc404:
            await c.bootstrap(WS, context_id=CTX)
    assert (exc404.value.code, exc404.value.details) == ("NOT-001", {"agent_id": "a1"})


@pytest.mark.parametrize(
    "challenge",
    [
        'Basic realm="proxy", Bearer error="insufficient_scope", scope="memory:write"',
        'Bearer error="insufficient_scope", scope="memory:write", Basic realm="proxy"',
        "Bearer error=insufficient_scope, scope=memory:write",
    ],
    ids=["after-another", "before-another", "bare-tokens"],
)
def test_scope_challenge_is_read_wherever_the_bearer_challenge_sits(challenge):
    """httpx joins repeated WWW-Authenticate headers with ", "."""
    err = _raised(403, {"error": "HTTP-403", "message": "x"}, **{"WWW-Authenticate": challenge})
    assert isinstance(err, KaguraAuthError)
    assert err.details == {"required_scope": "memory:write"}


def test_a_non_bearer_challenge_naming_the_scope_error_is_not_read():
    challenge = 'Basic realm="error=insufficient_scope", charset="UTF-8"'
    err = _raised(403, {"error": "HTTP-403", "message": "x"}, **{"WWW-Authenticate": challenge})
    assert type(err) is KaguraConnectionError


@pytest.mark.asyncio
async def test_list_tags_rest_422_keeps_the_rest_body_on_the_mcp_style_error():
    errors = [{"loc": ["query", "with_tags"], "msg": "Too many tags", "type": "value_error"}]
    body = {
        "error": "VAL-001",
        "message": "Request validation failed",
        "details": {"errors": errors},
    }
    client = KaguraClient(api_key="test", mcp_url="https://test.com/mcp")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_respond(422, body)))
    try:
        with pytest.raises(KaguraError) as exc:
            await client.list_tags(context_id=CTX, with_tags=["a"])
    finally:
        await client.close()
    # The MCP tool's code, as before; the fields are the REST body's.
    assert exc.value.code == "invalid_argument"
    assert exc.value.details == {"errors": errors}


# ---------------------------------------------------------------------------
# The error body is decoded once (#311)
# ---------------------------------------------------------------------------


class _CountingResponse(httpx.Response):
    """An ``httpx.Response`` that counts how often its body is decoded."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.decodes = 0

    def json(self, **kwargs: Any) -> Any:
        self.decodes += 1
        return super().json(**kwargs)


_SCOPE_ONLY = {"WWW-Authenticate": 'Bearer error="insufficient_scope", scope="memory:write"'}
_DB_002 = {"error": "DB-002", "message": "Database down", "details": {"reason": "pool"}}


async def _resource_tokens(c: ResourceClient) -> Any:
    return await c.list_tokens()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cls", "call", "status", "body", "headers", "raised"),
    [
        (ResourceClient, _resource_tokens, 403, _DENY_BODY, {}, KaguraConnectionError),
        (FilesClient, _files_list, 403, _DENY_BODY, {}, KaguraConnectionError),
        (WorkspaceClient, _workspace_members, 403, _DENY_BODY, {}, KaguraConnectionError),
        (SecretClient, _secret_pubkeys, 403, _DENY_BODY, {}, KaguraConnectionError),
        (ResourceClient, _resource_tokens, 403, _DENY_BODY, _SCOPE_ONLY, KaguraAuthError),
        (ResourceClient, _resource_tokens, 403, _CAPACITY_BODY, {}, KaguraCapacityLockedError),
        (ResourceClient, _resource_tokens, 404, _DENY_BODY, {}, KaguraNotFoundError),
        (WorkspaceClient, _workspace_members, 429, {"detail": "slow"}, {}, KaguraQuotaError),
        (SecretClient, _secret_pubkeys, 429, {"detail": "slow"}, {}, KaguraConnectionError),
        (ResourceClient, _resource_tokens, 503, _DB_002, {}, KaguraServiceUnavailableError),
        (ResourceClient, _resource_tokens, 409, _DENY_BODY, {}, KaguraConnectionError),
    ],
    ids=[
        "403-default-hook",
        "403-files-hook",
        "403-workspace-hook",
        "403-secret-hook",
        "403-scope",
        "403-capacity-gate",
        "404",
        "429-workspace-hook",
        "429-secret-hook",
        "503",
        "409",
    ],
)
async def test_rest_error_body_is_decoded_once(cls, call, status, body, headers, raised):
    """Gate, scope, hook wording and the code/details all read one decode."""
    sent: list[_CountingResponse] = []

    def handler(_req: httpx.Request) -> httpx.Response:
        sent.append(_CountingResponse(status, json=body, headers=headers))
        return sent[-1]

    async with _rest(cls, handler) as c:
        with pytest.raises(raised) as exc:
            await call(c)
    assert [r.decodes for r in sent] == [1]
    # The one decode still gives the error the body's code.
    assert (exc.value.code is not None) == ("error" in body)


@pytest.mark.parametrize(
    ("status", "body", "headers", "decodes"),
    [
        (401, {"error": "invalid_token"}, {}, 0),
        (403, _MCP_SCOPE_BODY, {"WWW-Authenticate": _MCP_SCOPE_CHALLENGE}, 1),
        (403, {"error": "access_denied", "error_description": "No."}, {}, 1),
        (429, {"detail": "slow"}, {}, 1),
        (503, {"error": "temporarily_unavailable", "error_description": "Retry."}, {}, 1),
        (500, {"detail": "boom"}, {}, 1),
    ],
    ids=["401-not-read", "403-scope", "403-other", "429", "503", "500"],
)
def test_raise_for_kagura_status_decodes_the_body_once(status, body, headers, decodes):
    request = httpx.Request("POST", "https://test.com/mcp")
    response = _CountingResponse(status, json=body, headers=headers, request=request)
    with pytest.raises(KaguraError):
        raise_for_kagura_status(
            httpx.HTTPStatusError(str(status), request=request, response=response)
        )
    assert response.decodes == decodes


async def _list_tags_via_rest(c: KaguraClient) -> Any:
    return await c.list_tags(context_id=CTX, with_tags=["a"])  # REST, mcp_tool="list_tags"


async def _server_info(c: KaguraClient) -> Any:
    return await c.get_server_info()  # REST, no mcp_tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "status", "body", "raised", "decodes"),
    [
        (_list_tags_via_rest, 422, {"error": "VAL-001", "message": "bad"}, KaguraError, 1),
        (_list_tags_via_rest, 404, {"detail": "no context"}, KaguraNotFoundError, 1),
        (_list_tags_via_rest, 500, {"detail": "boom"}, KaguraConnectionError, 1),
        (_server_info, 404, {"detail": "gone"}, KaguraConnectionError, 1),
        (_server_info, 401, {"error": "invalid_token"}, KaguraAuthError, 0),
    ],
    ids=["mcp-remap-422", "mcp-remap-404", "fall-through-500", "no-mcp-tool-404", "401-not-read"],
)
async def test_kagura_client_rest_error_body_is_decoded_once(call, status, body, raised, decodes):
    """KaguraClient's REST reads: the MCP-style remap and the status mapping share one decode."""
    sent: list[_CountingResponse] = []

    def handler(_req: httpx.Request) -> httpx.Response:
        sent.append(_CountingResponse(status, json=body))
        return sent[-1]

    async with kagura_client_on(handler) as client:
        with pytest.raises(raised):
            await call(client)
    assert [r.decodes for r in sent] == [decodes]


def test_error_body_decodes_lazily_once_and_shares_no_dict():
    """Nothing is decoded until a reader asks; then once, and no two readers share a dict."""
    body = {"error": "AUTH-101", "message": "No.", "details": {"gate": "plan", "n": [1]}}
    response = _CountingResponse(403, json=body)

    error = ErrorBody(response)
    assert response.decodes == 0

    assert error.envelope == ("AUTH-101", "No.", {"gate": "plan", "n": [1]})
    assert error.fields == ("AUTH-101", {"gate": "plan", "n": [1]})
    assert (error.detail, error.jsonrpc) == ("No.", None)
    assert response.decodes == 1
    # Each reader is computed once and kept.
    assert error.envelope is error.envelope
    # The envelope's details and the fields are copies: changing one leaves the other.
    envelope_details, fields = error.envelope[2], error.fields[1]
    assert envelope_details is not fields
    envelope_details["n"].append(2)
    assert fields == {"gate": "plan", "n": [1]}
    assert ErrorBody(_CountingResponse(403, json=body)).fields == ("AUTH-101", body["details"])


@pytest.mark.parametrize(
    "content",
    [b"", b"not json", b"\xff\xfe", b"null", b"[1]", b'"x"'],
    ids=["empty", "text", "bad-utf8", "null", "list", "string"],
)
def test_error_body_that_is_not_a_json_object_reads_as_nothing(content: bytes):
    error = ErrorBody(httpx.Response(500, content=content))
    assert (error.detail, error.envelope, error.fields, error.jsonrpc) == ("", None, None, None)
