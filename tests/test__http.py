"""Unit tests for ``kagura_memory._http`` helpers.

Focused on ``extract_detail``'s contract — the FastAPI validation-error
list path is the new behavior added in #110 so a bare ``HTTP 422`` no
longer hides which field actually failed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from kagura_memory._http import extract_detail, validate_https_url


def _response_with_json(payload: object) -> MagicMock:
    """Build a minimal httpx.Response-shaped mock with ``.json()`` returning ``payload``."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = payload
    return resp


def _response_with_bad_json() -> MagicMock:
    """Mock whose ``.json()`` raises ``ValueError`` — simulates a non-JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.side_effect = ValueError("not json")
    return resp


# ---------------------------------------------------------------------------
# str detail (legacy FastAPI HTTPException shape)
# ---------------------------------------------------------------------------


def test_string_detail_returned_as_is():
    resp = _response_with_json({"detail": "User not found"})
    assert extract_detail(resp) == "User not found"


def test_empty_string_detail_returns_empty():
    resp = _response_with_json({"detail": ""})
    assert extract_detail(resp) == ""


# ---------------------------------------------------------------------------
# list detail (FastAPI validation-error shape — 422)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detail, expected",
    [
        # The motivating case from #110: single 422 with loc=[body, workspace_id]
        pytest.param(
            [
                {
                    "type": "uuid_parsing",
                    "loc": ["body", "workspace_id"],
                    "msg": "Input should be a valid UUID",
                    "input": "auto",
                }
            ],
            "body.workspace_id: Input should be a valid UUID",
            id="loc_path",
        ),
        # FastAPI puts list indices in loc as ints — stringify so the path stays printable
        pytest.param(
            [{"loc": ["body", "items", 0, "name"], "msg": "Field required"}],
            "body.items.0.name: Field required",
            id="integer_index_in_loc",
        ),
        # msg-only entry (non-field validator) → return msg alone
        pytest.param(
            [{"msg": "Internal validation failure"}],
            "Internal validation failure",
            id="missing_loc",
        ),
        # loc: [] is treated the same as missing loc
        pytest.param(
            [{"loc": [], "msg": "Top-level validation failure"}],
            "Top-level validation failure",
            id="empty_loc",
        ),
    ],
)
def test_single_entry_loc_variants(detail: list[dict], expected: str):
    """Single-entry FastAPI 422 shapes: loc with strings, ints, missing, or empty."""
    assert extract_detail(_response_with_json({"detail": detail})) == expected


def test_multiple_validation_errors_joined_with_semicolon():
    """Multiple field failures are joined with ``"; "`` so a one-line CLI error stays one line."""
    resp = _response_with_json(
        {
            "detail": [
                {"loc": ["body", "workspace_id"], "msg": "Input should be a valid UUID"},
                {"loc": ["body", "size_bytes"], "msg": "Input should be greater than 0"},
            ]
        }
    )
    assert extract_detail(resp) == (
        "body.workspace_id: Input should be a valid UUID; "
        "body.size_bytes: Input should be greater than 0"
    )


def test_malformed_entries_skipped_well_formed_kept():
    """Silent skip on per-entry malformed data: a single bad entry must not blank the line."""
    resp = _response_with_json(
        {
            "detail": [
                "not a dict",
                {"loc": ["body", "x"]},
                {"msg": ""},
                42,
                {"loc": ["body", "y"], "msg": "Field required"},
            ]
        }
    )
    assert extract_detail(resp) == "body.y: Field required"


def test_empty_list_detail_returns_empty():
    resp = _response_with_json({"detail": []})
    assert extract_detail(resp) == ""


def test_list_with_only_malformed_entries_returns_empty():
    """If nothing in the list can be formatted, caller falls back to ``response.text``."""
    resp = _response_with_json({"detail": ["junk", 42, {}, {"loc": ["x"]}]})
    assert extract_detail(resp) == ""


# ---------------------------------------------------------------------------
# Non-JSON / non-dict / unsupported shapes
# ---------------------------------------------------------------------------


def test_non_json_body_returns_empty():
    resp = _response_with_bad_json()
    assert extract_detail(resp) == ""


def test_non_dict_body_returns_empty():
    resp = _response_with_json(["not", "a", "dict"])
    assert extract_detail(resp) == ""


def test_missing_detail_field_returns_empty():
    resp = _response_with_json({"error": "oops"})
    assert extract_detail(resp) == ""


def test_unsupported_detail_type_returns_empty():
    """``detail`` that is neither str nor list → empty (don't crash, don't guess)."""
    resp = _response_with_json({"detail": 42})
    assert extract_detail(resp) == ""


def test_unicode_decode_error_returns_empty():
    """``.json()`` raising ``UnicodeDecodeError`` (e.g. binary body) → empty."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.side_effect = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    assert extract_detail(resp) == ""


# ---------------------------------------------------------------------------
# validate_https_url — HTTPS enforcement with a localhost dev exception (#189)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "https://localhost",  # https always fine, localhost or not
        "https://localhost.evil.com",
    ],
)
def test_https_always_allowed(url: str):
    """Any https:// URL passes regardless of host."""
    validate_https_url(url)  # must not raise


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://localhost/",
        "http://localhost:8080",
        "http://localhost:8080/mcp",
        "http://localhost?ready=1",
        "http://127.0.0.1",
        "http://127.0.0.1:5000/path",
        "http://[::1]",
        "http://[::1]:9000/mcp",
    ],
)
def test_localhost_http_allowed(url: str):
    """Genuine loopback hosts (optionally with port/path/query) are allowed over plain HTTP."""
    validate_https_url(url)  # must not raise


@pytest.mark.parametrize(
    "url",
    [
        # The motivating bypass from #189: attacker host with a "localhost" prefix.
        "http://localhost.evil.com",
        "http://localhost.evil.com/steal",
        "http://127.0.0.1.evil.com",
        "http://[::1].evil.com",
        # userinfo trick — "localhost" appears before an "@" delimiting the real host.
        "http://localhost@evil.com",
        "http://127.0.0.1@evil.com",
        # plain remote host
        "http://evil.com",
    ],
)
def test_http_non_localhost_rejected(url: str):
    """Plain-HTTP URLs whose real host is not loopback must be rejected."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_https_url(url)


def test_reject_message_includes_label_and_url():
    """The error surfaces the caller-supplied label and the offending URL."""
    with pytest.raises(ValueError, match=r"MCP URL must use HTTPS.*localhost\.evil\.com"):
        validate_https_url("http://localhost.evil.com", label="MCP URL")


def test_retry_after_seconds_parses_digits_else_none():
    """_retry_after_seconds honors integer seconds, else None (incl. HTTP-date / absent)."""
    from kagura_memory._http import _retry_after_seconds

    class _Resp:
        def __init__(self, headers):
            self.headers = headers

    assert _retry_after_seconds(_Resp({"Retry-After": "30"})) == 30
    assert _retry_after_seconds(_Resp({"Retry-After": " 45 "})) == 45  # stripped
    assert _retry_after_seconds(_Resp({"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"})) is None
    assert _retry_after_seconds(_Resp({})) is None


# ---------------------------------------------------------------------------
# memory-cloud canonical envelope ({"error", "message", "details"})
# ---------------------------------------------------------------------------


def test_envelope_message_returned():
    resp = _response_with_json(
        {"error": "AUTH-101", "message": "Insufficient permissions", "details": {}}
    )
    assert extract_detail(resp) == "Insufficient permissions"


def test_envelope_validation_errors_appended():
    resp = _response_with_json(
        {
            "error": "VAL-001",
            "message": "Request validation failed",
            "details": {
                "errors": [{"loc": ["body", "role"], "msg": "Value error, role=owner", "type": "v"}]
            },
        }
    )
    assert extract_detail(resp) == "Request validation failed: body.role: Value error, role=owner"


def test_envelope_with_malformed_details_falls_back_to_message():
    resp = _response_with_json(
        {"error": "REQ-001", "message": "expires_days is required", "details": "oops"}
    )
    assert extract_detail(resp) == "expires_days is required"


def test_detail_takes_precedence_over_message():
    # A body carrying both shapes keeps the legacy FastAPI semantics.
    resp = _response_with_json({"detail": "from detail", "message": "from message"})
    assert extract_detail(resp) == "from detail"


def test_non_string_message_returns_empty():
    resp = _response_with_json({"error": "X", "message": 42, "details": {}})
    assert extract_detail(resp) == ""
