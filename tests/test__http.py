"""Unit tests for ``kagura_memory._http`` helpers.

Focused on ``extract_detail``'s contract — the FastAPI validation-error
list path is the new behavior added in #110 so a bare ``HTTP 422`` no
longer hides which field actually failed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from kagura_memory._http import extract_detail


def _response_with_json(payload: object) -> MagicMock:
    """Build a minimal httpx.Response-shaped mock with ``.json()`` returning ``payload``."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = payload
    return resp


def _response_with_bad_json() -> MagicMock:
    """Mock whose ``.json()`` raises — simulates a non-JSON / HTML / empty body."""
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


def test_single_validation_error_formats_with_loc_path():
    """The motivating case from #110: a single 422 with ``loc=[body, workspace_id]``."""
    resp = _response_with_json(
        {
            "detail": [
                {
                    "type": "uuid_parsing",
                    "loc": ["body", "workspace_id"],
                    "msg": "Input should be a valid UUID",
                    "input": "auto",
                }
            ]
        }
    )
    assert extract_detail(resp) == "body.workspace_id: Input should be a valid UUID"


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


def test_loc_with_integer_list_index_stringified():
    """FastAPI puts list indices in ``loc`` as ints — stringify so the path stays printable."""
    resp = _response_with_json(
        {
            "detail": [
                {"loc": ["body", "items", 0, "name"], "msg": "Field required"},
            ]
        }
    )
    assert extract_detail(resp) == "body.items.0.name: Field required"


def test_missing_loc_uses_msg_alone():
    """Entry with ``msg`` but no ``loc`` (e.g. a non-field validator) → return msg alone."""
    resp = _response_with_json(
        {
            "detail": [
                {"msg": "Internal validation failure"},
            ]
        }
    )
    assert extract_detail(resp) == "Internal validation failure"


def test_empty_loc_list_uses_msg_alone():
    """``loc: []`` is treated the same as missing ``loc``."""
    resp = _response_with_json(
        {
            "detail": [
                {"loc": [], "msg": "Top-level validation failure"},
            ]
        }
    )
    assert extract_detail(resp) == "Top-level validation failure"


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
