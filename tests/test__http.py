"""Unit tests for ``kagura_memory._http`` helpers.

Focused on ``extract_detail``'s contract — the FastAPI validation-error
list path is the new behavior added in #110 so a bare ``HTTP 422`` no
longer hides which field actually failed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from kagura_memory._http import (
    base_url_from_mcp,
    extract_detail,
    jsonrpc_error_body,
    mcp_session_expired,
    mcp_session_header,
    mcp_url_guardrails_off,
    mcp_url_has_tools_allowlist,
    mcp_url_with_query,
    mcp_url_without_query,
    normalize_guardrails,
    normalize_uuid,
    validate_https_url,
    validate_lat_lon,
)
from tests.conftest import SESSION_EXPIRED_BODY


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
# JSON-RPC error bodies (the MCP transport's 4xx shape, #252)
# ---------------------------------------------------------------------------


def test_jsonrpc_error_message_is_the_detail():
    """The MCP transport's 4xx carries a JSON-RPC ``error`` object, not ``detail``.

    Without this a KaguraClient surfaced the session-expired 404 as a bare
    ``Client error '404 Not Found'`` instead of the server's own message.
    """
    resp = _response_with_json(SESSION_EXPIRED_BODY)
    assert extract_detail(resp) == (
        "MCP session not found or expired. Please re-initialize your connection."
    )


def test_jsonrpc_error_without_string_message_returns_empty():
    resp = _response_with_json({"jsonrpc": "2.0", "error": {"code": -32600}, "id": None})
    assert extract_detail(resp) == ""


def test_oauth_style_error_description_is_the_detail():
    """The MCP transport's workspace-URL 400/403 body is OAuth-style, not ``detail``.

    Without this a KaguraClient on a ``/mcp/w/{id}`` URL surfaced a 403 as a
    bare ``Client error '403 Forbidden'`` with the server's reason dropped.
    """
    resp = _response_with_json(
        {"error": "access_denied", "error_description": "You are not a member of this workspace."}
    )
    assert extract_detail(resp) == "You are not a member of this workspace."


def test_jsonrpc_error_body_returns_the_envelope():
    assert jsonrpc_error_body(_response_with_json(SESSION_EXPIRED_BODY)) == SESSION_EXPIRED_BODY


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "invalid_token", "error_description": "expired"},  # the 401 shape
        {"detail": "nope"},
        ["not", "a", "dict"],
    ],
)
def test_jsonrpc_error_body_rejects_other_shapes(payload: object):
    assert jsonrpc_error_body(_response_with_json(payload)) is None


def test_jsonrpc_error_body_non_json_returns_none():
    assert jsonrpc_error_body(_response_with_bad_json()) is None


def test_mcp_session_header():
    assert mcp_session_header("sess-1") == {"mcp-session-id": "sess-1"}
    assert mcp_session_header(None) == {}


def test_mcp_session_expired_on_404_for_a_session_request():
    resp = httpx.Response(404, json=SESSION_EXPIRED_BODY)
    assert mcp_session_expired(resp, "sess-1") is True


def test_mcp_session_expired_on_plain_404():
    """A 404 on a session-carrying request means "re-initialize" (MCP Streamable HTTP)."""
    assert mcp_session_expired(httpx.Response(404, text="Not Found"), "sess-1") is True


def test_mcp_session_not_expired_without_a_session_id():
    """A request that carried no session cannot have lost one — never re-initialize for it."""
    assert mcp_session_expired(httpx.Response(404, json=SESSION_EXPIRED_BODY), None) is False


@pytest.mark.parametrize("status", [200, 400, 401, 500])
def test_mcp_session_not_expired_on_other_statuses(status: int):
    assert mcp_session_expired(httpx.Response(status, json=SESSION_EXPIRED_BODY), "s") is False


def test_mcp_session_not_expired_on_modern_method_not_found():
    """memory-cloud answers a modern (2026-07-28) unknown method with 404 + -32601 (#1544).

    That path is stateless and ignores the session id, so re-initializing
    would only mint an orphan session and replay the same 404.
    """
    body = {"jsonrpc": "2.0", "id": 5, "error": {"code": -32601, "message": "Method not found"}}
    assert mcp_session_expired(httpx.Response(404, json=body), "sess-1") is False


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


@pytest.mark.parametrize(
    "url",
    [
        # Every URL parser reads the scheme without regard to case (#274).
        "HTTP://evil.com",
        "Http://evil.com/mcp",
        "hTtP://evil.com",
        # A URL parser drops surrounding whitespace before reading the scheme.
        " http://evil.com/mcp",
        "http://evil.com/mcp\n",
        "\thttp://evil.com",
        "\u3000http://evil.com",
        # WHATWG parsers (Node, Rust's url crate) also drop C0 controls around the
        # URL and a tab or newline anywhere in it ...
        "\x00http://evil.com",
        "\x1fhttp://evil.com",
        "ht\ttp://evil.com",
        "http\n://evil.com",
        # ... and read a special scheme without its slashes as http://host.
        "http:/evil.com",
        "http:evil.com",
        "http:\\\\evil.com",
        # The loopback exception, whatever the case, still needs a loopback host.
        "HTTP://LOCALHOST.evil.com",
        "HTTP://localhost@evil.com",
        " http://127.0.0.1.evil.com",
        # ASCII-only: no Unicode case fold spells "localhost" ("ſ" folds to "s").
        "http://localhoſt",
    ],
)
def test_plain_http_in_any_spelling_rejected(url: str):
    """No spelling a URL parser reads as plain HTTP to a remote host gets past the check."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_https_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "HTTPS://api.example.com",
        "hTtPs://api.example.com/mcp",
        " https://api.example.com/mcp\n",
        "HTTP://LOCALHOST:8080/mcp",
        " http://127.0.0.1:5000/path",
        "Http://[::1]:9000/mcp",
    ],
)
def test_scheme_case_and_surrounding_whitespace_do_not_matter(url: str):
    """HTTPS in any case passes, and so does loopback HTTP in any case."""
    validate_https_url(url)  # must not raise


def test_reject_message_shows_the_url_as_a_parser_reads_it():
    """The message names the trimmed URL, so the rejected scheme is visible."""
    with pytest.raises(ValueError, match=r"\(got: HTTP://evil\.com/mcp\)"):
        validate_https_url("  HTTP://evil.com/mcp\n", label="MCP URL")


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
# normalize_uuid — shared canonicalize-before-URL-interpolation guard
# ---------------------------------------------------------------------------

_CANONICAL = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.mark.parametrize(
    "spelling",
    [
        _CANONICAL,  # canonical passes through unchanged
        "{" + _CANONICAL + "}",  # braces
        _CANONICAL.replace("-", ""),  # dashless 32-hex
        f"urn:uuid:{_CANONICAL}",  # urn prefix
        _CANONICAL.upper(),  # uppercase → lowercased canonical
    ],
)
def test_normalize_uuid_canonicalizes_tolerated_spellings(spelling: str):
    """Every spelling ``uuid.UUID`` tolerates must come out canonical."""
    assert normalize_uuid(spelling, label="agent_id") == _CANONICAL


@pytest.mark.parametrize("bad", ["not-a-uuid", "", "../../admin", None, 42])
def test_normalize_uuid_rejects_non_uuids_with_label(bad):
    """Garbage (including non-str runtime values) raises with the caller's label."""
    with pytest.raises(ValueError, match="workspace_id must be a UUID"):
        normalize_uuid(bad, label="workspace_id")


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


# ---------------------------------------------------------------------------
# validate_lat_lon — one coordinate rule shared by recall_nearby + CLI --location
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (0.0, 0.0),
        (35.68, 139.76),
        (90.0, 180.0),  # poles / antimeridian are valid points, not errors
        (-90.0, -180.0),
        (0, 0),  # ints are numbers too
    ],
)
def test_validate_lat_lon_accepts_valid_points(lat: float, lon: float):
    """In-range coordinates, including the boundaries, must not raise."""
    validate_lat_lon(lat, lon)


@pytest.mark.parametrize(
    ("lat", "lon", "expected"),
    [
        (91.0, 0.0, "lat"),
        (-91.0, 0.0, "lat"),
        (0.0, 181.0, "lon"),
        (0.0, -181.0, "lon"),
    ],
)
def test_validate_lat_lon_rejects_out_of_range(lat: float, lon: float, expected: str):
    """Out-of-range coordinates raise, naming the offending axis."""
    with pytest.raises(ValueError, match=expected):
        validate_lat_lon(lat, lon)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (float("inf"), 0.0),
        (0.0, float("-inf")),
    ],
)
def test_validate_lat_lon_rejects_nan_and_inf(lat: float, lon: float):
    """NaN/inf must not reach the server as a coordinate.

    This depends on the ``not -90 <= lat <= 90`` form: every comparison against
    NaN is False, so the chain is False and ``not`` makes it raise. The
    equivalent-looking ``lat < -90 or lat > 90`` would let NaN through — do not
    "simplify" the guard into that shape.
    """
    with pytest.raises(ValueError, match="lat|lon"):
        validate_lat_lon(lat, lon)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        ("35.68", 139.76),  # the exact footgun the server 422s on
        (35.68, "139.76"),
        (None, 0.0),
        (0.0, None),
        ([35.68], 0.0),
        (True, 0.0),  # bool is an int subclass, but never a coordinate
    ],
)
def test_validate_lat_lon_rejects_non_numeric(lat: object, lon: object):
    """Non-numeric coordinates raise ValueError, not an opaque TypeError.

    A stringified coordinate must fail here rather than on the wire: coercing
    ``"35.68"`` would only move the server's 422 somewhere less debuggable.
    """
    with pytest.raises(ValueError, match="must be a number"):
        validate_lat_lon(lat, lon)


# ---------------------------------------------------------------------------
# base_url_from_mcp / mcp_url_with_query / normalize_guardrails (#258)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mcp_url", "expected"),
    [
        ("https://host.example/mcp", "https://host.example"),
        ("https://host.example/mcp/w/ws-1", "https://host.example"),
        ("https://host.example/mcp?guardrails=off", "https://host.example"),
        ("https://host.example/mcp/?profile=core", "https://host.example"),
        ("https://host.example/mcp/w/ws-1?guardrails=off&profile=core", "https://host.example"),
        ("https://host.example/mcp#frag", "https://host.example"),
        ("https://host.example/api?x=1", "https://host.example/api"),
    ],
)
def test_base_url_from_mcp_drops_query_and_fragment(mcp_url: str, expected: str):
    """A query after a bare ``/mcp`` must not leak into the REST base URL."""
    assert base_url_from_mcp(mcp_url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://h/mcp?tools=recall,remember", True),
        ("https://h/mcp?profile=core&tools=", True),  # present, even empty
        ("https://h/mcp?profile=core", False),
        ("https://h/mcp?x=tools", False),
        ("https://h/mcp", False),
    ],
)
def test_mcp_url_has_tools_allowlist(url: str, expected: bool):
    assert mcp_url_has_tools_allowlist(url) is expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://h/mcp?guardrails=off", True),
        ("https://h/mcp?profile=core&guardrails=%20OFF", True),
        ("https://h/mcp?guardrails=off&guardrails=x", True),
        ("https://h/mcp?guardrails=x&guardrails=off", False),  # the server reads the first
        ("https://h/mcp?guardrails=offline", False),
        ("https://h/mcp?x=off", False),
        ("https://h/mcp", False),
    ],
)
def test_mcp_url_guardrails_off(url: str, expected: bool):
    assert mcp_url_guardrails_off(url) is expected


def test_mcp_url_with_query_sets_both_keys_in_order():
    url = mcp_url_with_query("https://h.example/mcp", guardrails="off", tool_profile="core")
    assert url == "https://h.example/mcp?guardrails=off&profile=core"


def test_mcp_url_with_query_keeps_existing_query_verbatim():
    url = mcp_url_with_query("https://h.example/mcp/w/ws?tools=recall,remember", guardrails="off")
    assert url == "https://h.example/mcp/w/ws?tools=recall,remember&guardrails=off"


def test_mcp_url_with_query_replaces_every_earlier_value():
    """The server reads only the first value, so an old one must not survive."""
    ctx = "11111111-2222-3333-4444-555555555555"
    url = mcp_url_with_query(
        f"https://h.example/mcp?guardrails={ctx}&x=1&guardrails=off&profile=full",
        guardrails="off",
        tool_profile="core",
    )
    assert url == "https://h.example/mcp?x=1&guardrails=off&profile=core"


def test_mcp_url_with_query_none_is_a_no_op():
    url = "https://h.example/mcp?guardrails=off"
    assert mcp_url_with_query(url) == url
    assert mcp_url_with_query(url, tool_profile="core") == url + "&profile=core"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://h.example/mcp?guardrails=a&tools=x,y&guardrails=off#f",
            "https://h.example/mcp?tools=x,y#f",
        ),
        # Names compare decoded, as the server reads them.
        ("https://h.example/mcp?guard%72ails=a&profile=core", "https://h.example/mcp?profile=core"),
        ("https://h.example/mcp?guardrails", "https://h.example/mcp"),
    ],
)
def test_mcp_url_without_query_drops_every_value(url, expected):
    assert mcp_url_without_query(url, "guardrails") == expected


@pytest.mark.parametrize(
    "url", ["https://h.example/mcp", "https://h.example/mcp?", "https://h.example/mcp?&tools=a,b&"]
)
def test_mcp_url_without_query_leaves_a_url_without_the_key_as_it_is(url):
    assert mcp_url_without_query(url, "guardrails") == url


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("off", "off"),
        (" OFF ", "off"),
        ("11111111-2222-3333-4444-555555555555", "11111111-2222-3333-4444-555555555555"),
        ("{11111111-2222-3333-4444-555555555555}", "11111111-2222-3333-4444-555555555555"),
    ],
)
def test_normalize_guardrails_accepts_off_or_uuid(value: str, expected: str):
    assert normalize_guardrails(value) == expected


@pytest.mark.parametrize("bad", ["", "on", "false", "my-context", "off,on"])
def test_normalize_guardrails_rejects_everything_else(bad: str):
    """The server silently ignores these, so they must fail loudly here."""
    with pytest.raises(ValueError, match="'off' or a context UUID"):
        normalize_guardrails(bad)
