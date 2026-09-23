"""Tests for ``kagura auth login --invite`` (issue #259).

Covers the pure helpers in :mod:`kagura_memory.auth.device_flow`
(``parse_invite``, ``build_invite_link``, ``invite_support``,
``fetch_system_info``) and the CLI prompt built on top of them.

The single-link ``/join`` → ``/device`` hand-off needs memory-cloud#1655,
which no released server ships yet, so
``JOIN_RETURN_TO_MIN_SERVER_VERSION`` is ``None`` and every server gets the
two-step fallback. Tests that exercise the single link patch that constant.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from click.testing import CliRunner

from kagura_memory.auth import device_flow
from kagura_memory.auth.credentials import reset_state_cache
from kagura_memory.auth.device_flow import (
    DeviceAuthorizationResponse,
    TokenResponse,
    build_invite_link,
    check_invite_origin,
    fetch_system_info,
    invite_support,
    parse_invite,
)
from kagura_memory.cli import main
from kagura_memory.exceptions import KaguraAuthExpiredError

# A valid token (matches ^[A-Za-z0-9_-]{20,128}$) that is easy to grep for.
SENTINEL = "SENTINEL_invite_tok_0123456789"
WEB = "https://app.example.com"
API = "https://api.example.com"
DEVICE_URI = f"{WEB}/device"
DEVICE_URI_COMPLETE = f"{WEB}/device?user_code=ABCD-1234"
HAND_OFF_VERSION = (0, 76, 0)
# make_oauth_client's default. Not httpx's own 5 s default, which would hide
# a /system/info probe that forgot its short timeout.
_OAUTH_TIMEOUT = 30.0


@pytest.fixture
def patched_default_path(tmp_path: Path, monkeypatch):
    """Redirect every credentials.py caller to a tmp_path credentials.json."""
    fake_path = tmp_path / ".kagura" / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.setattr("kagura_memory.auth.cli.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    reset_state_cache()
    yield fake_path
    reset_state_cache()


@pytest.fixture
def hand_off_released(monkeypatch):
    """Pretend memory-cloud#1655 shipped in v0.76.0."""
    monkeypatch.setattr(
        "kagura_memory.auth.device_flow.JOIN_RETURN_TO_MIN_SERVER_VERSION", HAND_OFF_VERSION
    )


def _device(
    verification_uri: str = DEVICE_URI,
    verification_uri_complete: str = DEVICE_URI_COMPLETE,
    expires_in: int = 600,
) -> DeviceAuthorizationResponse:
    return DeviceAuthorizationResponse(
        device_code="dc-1",
        user_code="ABCD-1234",
        verification_uri=verification_uri,
        verification_uri_complete=verification_uri_complete,
        expires_in=expires_in,
        interval=5,
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
    )


def _token() -> TokenResponse:
    return TokenResponse(
        access_token="atok-fresh",
        refresh_token="rtok-fresh",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:read memory:write",
        user_email="new@example.com",
        workspace_id="ws-abcdefgh",
        workspace_name="new-workspace",
    )


def _system_info(version: str = "0.76.0", *, beta_invites: bool | None = True) -> dict:
    features: dict = {"neural_memory": False}
    if beta_invites is not None:
        features["beta_invites"] = beta_invites
    return {"name": "Kagura Memory Cloud", "version": version, "features": features}


class _Server:
    """``httpx.MockTransport`` handler serving ``/api/v1/system/info``.

    ``info`` is the JSON body; ``status`` overrides the HTTP status; ``exc``
    is raised instead of answering (timeouts). Every request is recorded.
    """

    def __init__(
        self,
        info: dict | None = None,
        *,
        status: int = 200,
        exc: Callable[[httpx.Request], Exception] | None = None,
    ) -> None:
        self.info = info if info is not None else _system_info()
        self.status = status
        self.exc = exc
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.exc is not None:
            raise self.exc(request)
        if request.url.path != "/api/v1/system/info":
            return httpx.Response(404, json={"detail": "Not Found"})
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "Not Found"})
        return httpx.Response(200, json=self.info)

    def client_factory(self) -> Callable[..., httpx.AsyncClient]:
        return lambda *_a, **_k: httpx.AsyncClient(
            transport=httpx.MockTransport(self), timeout=_OAUTH_TIMEOUT
        )


def _invoke(
    args: list[str],
    server: _Server,
    *,
    device: DeviceAuthorizationResponse | None = None,
    poll: AsyncMock | None = None,
    browser_ok: bool = True,
):
    """Run ``kagura auth login`` with the device flow + browser stubbed."""
    poll = poll or AsyncMock(return_value=_token())
    authorize = AsyncMock(return_value=device or _device())
    with (
        patch("kagura_memory.auth.cli.make_oauth_client", server.client_factory()),
        patch("kagura_memory.auth.cli.authorize_device", authorize),
        patch("kagura_memory.auth.cli.poll_for_token", poll),
        patch("kagura_memory.auth.cli._try_open_browser", return_value=browser_ok) as browser,
    ):
        result = CliRunner().invoke(main, ["auth", "login", *args])
    return result, authorize, poll, browser


# ---------------------------------------------------------------------------
# parse_invite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "origin"),
    [
        (f"{WEB}/join/{SENTINEL}", WEB),
        (f"{WEB}/join/{SENTINEL}?utm_source=mail&x=1", WEB),
        (f"{WEB}/join/{SENTINEL}#frag", WEB),
        (f"{WEB}/join/{SENTINEL}?a=b#frag", WEB),
        (f"{WEB}/join/{SENTINEL}/", WEB),
        (f"{WEB}/kagura/app/join/{SENTINEL}", WEB),  # frontend under a base path
        (f"HTTPS://App.Example.com/join/{SENTINEL}", WEB),  # origin is case-normalised
        (f"http://localhost:3000/join/{SENTINEL}", "http://localhost:3000"),
        (f"http://[::1]:3000/join/{SENTINEL}", "http://[::1]:3000"),
        (f"https://app.example.com:443/join/{SENTINEL}", WEB),  # default port dropped
        (f"http://localhost:80/join/{SENTINEL}", "http://localhost"),
        (f"https://app.example.com:80/join/{SENTINEL}", "https://app.example.com:80"),
        (SENTINEL, None),
        (f"  {SENTINEL}  ", None),
    ],
)
def test_parse_invite_accepts_links_and_bare_tokens(value: str, origin: str | None):
    ref = parse_invite(value)
    assert ref.token == SENTINEL
    assert ref.origin == origin


def test_parse_invite_link_drops_query_and_fragment():
    ref = parse_invite(f"{WEB}/app/join/{SENTINEL}?utm=1#frag")
    assert ref.link == f"{WEB}/app/join/{SENTINEL}"


def test_parse_invite_bare_token_has_no_link():
    assert parse_invite(SENTINEL).link is None


def test_invite_ref_repr_hides_the_token():
    ref = parse_invite(f"{WEB}/join/{SENTINEL}")
    assert SENTINEL not in repr(ref)
    assert SENTINEL not in str(ref)


_BAD_INVITES = [
    pytest.param("", id="empty"),
    pytest.param("SENTINEL_short_tok", id="short-token"),  # 18 chars
    pytest.param("S" * 129, id="long-token"),
    pytest.param("SENTINEL_bad!chars_0123456789", id="bad-chars"),
    pytest.param("SENTINEL_bad.chars_0123456789", id="dot"),
    pytest.param("SENTINEL_invite\ntok_0123456789", id="embedded-newline"),
    pytest.param(f"{WEB}/invite/{SENTINEL}", id="wrong-path"),
    pytest.param(f"{WEB}/join/{SENTINEL}/extra", id="join-not-last"),
    pytest.param(f"{WEB}/join", id="join-without-token"),
    pytest.param(f"{WEB}/{SENTINEL}", id="no-join-segment"),
    pytest.param(f"{WEB}/join/SENTINEL_short_tok", id="link-short-token"),
    pytest.param(f"http://evil.example.com/join/{SENTINEL}", id="plain-http"),
    pytest.param(f"ftp://app.example.com/join/{SENTINEL}", id="ftp"),
    pytest.param(f"https:///join/{SENTINEL}", id="no-host"),
    pytest.param(f"https://app.example.com:99999/join/{SENTINEL}", id="bad-port"),
    pytest.param(f"app.example.com/join/{SENTINEL}", id="no-scheme"),
]


@pytest.mark.parametrize("value", _BAD_INVITES)
def test_parse_invite_rejects_malformed_values_without_echoing_them(value: str):
    with pytest.raises(ValueError) as excinfo:
        parse_invite(value)
    message = str(excinfo.value)
    assert "SENTINEL" not in message
    if value.strip():
        assert value.strip() not in message


# ---------------------------------------------------------------------------
# build_invite_link
# ---------------------------------------------------------------------------


def _return_to(link: str) -> str:
    [value] = parse_qs(urlsplit(link).query, strict_parsing=True)["return_to"]
    return value


def test_build_invite_link_uses_verification_uri_origin():
    link = build_invite_link(DEVICE_URI, DEVICE_URI_COMPLETE, SENTINEL)
    assert link is not None
    assert link.startswith(f"{WEB}/join/{SENTINEL}?return_to=")
    # return_to is fully percent-encoded (safe=""), so the query has one key.
    assert link == f"{WEB}/join/{SENTINEL}?return_to=%2Fdevice%3Fuser_code%3DABCD-1234"


def test_build_invite_link_return_to_round_trips_path_and_query():
    complete = f"{WEB}/device?user_code=ABCD-1234&lang=ja"
    link = build_invite_link(DEVICE_URI, complete, SENTINEL)
    assert link is not None
    parts = urlsplit(complete)
    assert _return_to(link) == f"{parts.path}?{parts.query}"


def test_build_invite_link_complete_uri_without_query():
    # authorize_device falls back to verification_uri when the server omits
    # verification_uri_complete, so return_to can be a bare path.
    link = build_invite_link(DEVICE_URI, DEVICE_URI, SENTINEL)
    assert link is not None
    assert _return_to(link) == "/device"


@pytest.mark.parametrize("token", ["short", "../../admin_0123456789abc", ""])
def test_build_invite_link_rejects_a_malformed_token(token: str):
    with pytest.raises(ValueError, match="invite token must be"):
        build_invite_link(DEVICE_URI, DEVICE_URI_COMPLETE, token)


def test_build_invite_link_keeps_frontend_base_path():
    link = build_invite_link(
        f"{WEB}/kagura/device", f"{WEB}/kagura/device?user_code=ABCD-1234", SENTINEL
    )
    assert link is not None
    assert link.startswith(f"{WEB}/kagura/join/{SENTINEL}?")
    assert _return_to(link) == "/kagura/device?user_code=ABCD-1234"


def test_build_invite_link_tolerates_trailing_slash_on_device():
    link = build_invite_link(f"{WEB}/device/", DEVICE_URI_COMPLETE, SENTINEL)
    assert link is not None
    assert link.startswith(f"{WEB}/join/{SENTINEL}?")


def test_build_invite_link_treats_the_default_port_as_the_same_origin():
    link = build_invite_link(f"{WEB}:443/device", DEVICE_URI_COMPLETE, SENTINEL)
    assert link == f"{WEB}/join/{SENTINEL}?return_to=%2Fdevice%3Fuser_code%3DABCD-1234"


def test_build_invite_link_allows_localhost_http():
    link = build_invite_link(
        "http://localhost:3000/device",
        "http://localhost:3000/device?user_code=ABCD-1234",
        SENTINEL,
    )
    assert link is not None
    assert link.startswith(f"http://localhost:3000/join/{SENTINEL}?return_to=")


@pytest.mark.parametrize(
    ("verification_uri", "complete"),
    [
        (f"{WEB}/activate", f"{WEB}/activate?user_code=ABCD-1234"),  # no /device
        (f"{WEB}/mydevice", f"{WEB}/mydevice?user_code=ABCD-1234"),  # not a segment
        (f"{WEB}/device/extra", f"{WEB}/device/extra?user_code=ABCD-1234"),
        ("http://app.example.com/device", "http://app.example.com/device?user_code=X"),
        ("javascript:alert(1)/device", "javascript:alert(1)/device"),
        (DEVICE_URI, "https://other.example.com/device?user_code=ABCD-1234"),
    ],
)
def test_build_invite_link_returns_none_when_join_cannot_be_placed(
    verification_uri: str, complete: str
):
    assert build_invite_link(verification_uri, complete, SENTINEL) is None


# ---------------------------------------------------------------------------
# check_invite_origin
# ---------------------------------------------------------------------------


def test_check_invite_origin_passes_bare_tokens_and_same_origin_links():
    check_invite_origin(parse_invite(SENTINEL), "https://anything.example.net/device")
    check_invite_origin(parse_invite(f"{WEB}/app/join/{SENTINEL}"), DEVICE_URI)
    check_invite_origin(parse_invite(f"https://APP.example.com/join/{SENTINEL}"), DEVICE_URI)
    # A spelled-out default port is the same origin, as in a browser.
    check_invite_origin(parse_invite(f"{WEB}:443/join/{SENTINEL}"), DEVICE_URI)
    check_invite_origin(parse_invite(f"{WEB}/join/{SENTINEL}"), f"{WEB}:443/device")


@pytest.mark.parametrize(
    ("verification_uri", "named"),
    [
        ("https://other.example.org/device", "https://other.example.org"),
        ("https://app.example.com:8443/device", "https://app.example.com:8443"),
        ("not a url", "not a url"),  # no parseable origin: named verbatim
    ],
)
def test_check_invite_origin_rejects_another_origin(verification_uri: str, named: str):
    with pytest.raises(ValueError, match="different server") as excinfo:
        check_invite_origin(parse_invite(f"{WEB}/join/{SENTINEL}"), verification_uri)
    assert f"({WEB})" in str(excinfo.value)
    assert f"({named})" in str(excinfo.value)
    assert SENTINEL not in str(excinfo.value)


# ---------------------------------------------------------------------------
# invite_support
# ---------------------------------------------------------------------------


def test_hand_off_version_is_unset_until_memory_cloud_1655_ships():
    assert device_flow.JOIN_RETURN_TO_MIN_SERVER_VERSION is None


def test_invite_support_never_hands_off_while_the_constant_is_unset():
    assert invite_support(_system_info("99.0.0")) == "two_step"


def test_invite_support_hands_off_at_or_after_the_release(hand_off_released):
    assert invite_support(_system_info("0.76.0")) == "hand_off"
    assert invite_support(_system_info("0.77.3")) == "hand_off"
    assert invite_support(_system_info("v1.0.0")) == "hand_off"


def test_invite_support_older_version_falls_back(hand_off_released):
    assert invite_support(_system_info("0.75.9")) == "two_step"


@pytest.mark.parametrize("version", ["", "dev", "0.76", "latest", None, 76])
def test_invite_support_unparseable_version_falls_back(hand_off_released, version):
    info = _system_info()
    info["version"] = version
    assert invite_support(info) == "two_step"


def test_invite_support_no_info_falls_back(hand_off_released):
    assert invite_support(None) == "two_step"


def test_invite_support_beta_invites_false_disables(hand_off_released):
    assert invite_support(_system_info("0.76.0", beta_invites=False)) == "disabled"


@pytest.mark.parametrize("flag", [None, "true", 1, {}], ids=["missing", "str", "int", "obj"])
def test_invite_support_beta_invites_not_true_disables(hand_off_released, flag):
    # memory-cloud reads a missing flag as off, and a server older than the
    # flag (before v0.70.0) has no /join route at all.
    info = _system_info("0.76.0", beta_invites=None)
    if flag is not None:
        info["features"]["beta_invites"] = flag
    assert invite_support(info) == "disabled"


def test_invite_support_without_a_features_object_is_unknown(hand_off_released):
    # No features object says nothing about invites: fall through to the version.
    assert invite_support({"version": "0.76.0"}) == "hand_off"
    assert invite_support({"version": "0.76.0", "features": "weird"}) == "hand_off"
    assert invite_support({"version": "0.75.0", "features": None}) == "two_step"


# ---------------------------------------------------------------------------
# fetch_system_info
# ---------------------------------------------------------------------------


async def _fetch(handler: Callable[[httpx.Request], httpx.Response]) -> dict | None:
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, timeout=_OAUTH_TIMEOUT) as client:
        return await fetch_system_info(client, f"{API}/")


@pytest.mark.asyncio
async def test_fetch_system_info_returns_raw_json():
    server = _Server(_system_info("0.75.0", beta_invites=False))
    info = await _fetch(server)
    assert info == _system_info("0.75.0", beta_invites=False)
    [request] = server.requests
    assert str(request.url) == f"{API}/api/v1/system/info"
    assert request.method == "GET"
    assert "authorization" not in request.headers
    # The device code is already ticking: the probe overrides the client's 30 s.
    assert device_flow._SYSTEM_INFO_TIMEOUT_SEC == 5.0
    assert request.extensions["timeout"] == dict.fromkeys(("connect", "read", "write", "pool"), 5.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda r: httpx.Response(404, json={"detail": "Not Found"}), id="404"),
        pytest.param(lambda r: httpx.Response(500, text="boom"), id="500"),
        pytest.param(lambda r: httpx.Response(200, text="<html>"), id="not-json"),
        pytest.param(lambda r: httpx.Response(200, json=["x"]), id="not-object"),
    ],
)
async def test_fetch_system_info_returns_none_on_bad_responses(handler):
    assert await _fetch(handler) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(lambda r: httpx.ReadTimeout("timed out", request=r), id="timeout"),
        pytest.param(lambda r: httpx.ConnectError("refused", request=r), id="connect"),
    ],
)
async def test_fetch_system_info_returns_none_on_network_errors(exc):
    assert await _fetch(_Server(exc=exc)) is None


# ---------------------------------------------------------------------------
# CLI: argument validation (no network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", _BAD_INVITES)
def test_login_bad_invite_is_usage_error_before_any_network_call(
    value: str, patched_default_path: Path
):
    server = _Server()
    factory = server.client_factory()
    with (
        patch("kagura_memory.auth.cli.make_oauth_client", side_effect=factory) as make_client,
        patch("kagura_memory.auth.cli.authorize_device", new_callable=AsyncMock) as authorize,
    ):
        result = CliRunner().invoke(main, ["auth", "login", "--invite", value])
    assert result.exit_code == 2, result.output  # click.UsageError
    assert "--invite" in result.output
    assert "SENTINEL" not in result.output
    make_client.assert_not_called()
    authorize.assert_not_called()
    assert server.requests == []
    assert not patched_default_path.exists()


def test_login_without_invite_does_not_probe_system_info(patched_default_path: Path):
    server = _Server()
    result, _, _, browser = _invoke([], server)
    assert result.exit_code == 0, result.output
    assert server.requests == []
    assert "/join/" not in result.output
    browser.assert_called_once_with(DEVICE_URI_COMPLETE)


# ---------------------------------------------------------------------------
# CLI: single-link hand-off (constant patched)
# ---------------------------------------------------------------------------


def _prompt_block(output: str) -> list[str]:
    """Lines from the one-time-code line up to the first blank line after it."""
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if "one-time code" in line)
    block: list[str] = []
    for line in lines[start:]:
        if not line.strip():
            break
        block.append(line)
    return block


def test_login_invite_single_link(hand_off_released, patched_default_path: Path):
    server = _Server(_system_info("0.76.0"))
    result, _, poll, browser = _invoke(["--invite", f"{WEB}/join/{SENTINEL}"], server)
    assert result.exit_code == 0, result.output

    link = f"{WEB}/join/{SENTINEL}?return_to=%2Fdevice%3Fuser_code%3DABCD-1234"
    block = _prompt_block(result.output)
    assert block[0].strip() == "! First copy your one-time code: ABCD-1234"
    assert "accept your invite" in result.output
    assert f"    {link}" in block
    assert "approval page with the code filled in" in result.output
    # The single-link prompt always ends with the plain verification_uri_complete.
    assert block[-1].strip() == DEVICE_URI_COMPLETE
    browser.assert_called_once_with(link)
    poll.assert_awaited_once()


def test_login_invite_link_host_comes_from_verification_uri_not_server(
    hand_off_released, patched_default_path: Path
):
    """API on api.<host>, web app on <host>: the link lands on the web app."""
    server = _Server(_system_info("0.76.0"))
    result, authorize, _, browser = _invoke(["--server", API, "--invite", SENTINEL], server)
    assert result.exit_code == 0, result.output
    assert authorize.call_args.args[1] == API
    [probe] = server.requests
    assert str(probe.url) == f"{API}/api/v1/system/info"
    assert probe.extensions["timeout"]["read"] == 5.0
    [opened] = browser.call_args.args
    assert opened.startswith(f"{WEB}/join/{SENTINEL}?return_to=")
    assert API not in opened


def test_login_invite_no_browser_prints_but_does_not_open(
    hand_off_released, patched_default_path: Path
):
    server = _Server(_system_info("0.76.0"))
    result, _, _, browser = _invoke(["--invite", SENTINEL, "--no-browser"], server)
    assert result.exit_code == 0, result.output
    browser.assert_not_called()
    assert f"{WEB}/join/{SENTINEL}?return_to=" in result.output
    assert "--no-browser" in result.output
    assert _prompt_block(result.output)[-1].strip() == DEVICE_URI_COMPLETE


def test_login_invite_browser_failure_prints_manual_hint(
    hand_off_released, patched_default_path: Path
):
    server = _Server(_system_info("0.76.0"))
    result, _, _, _ = _invoke(["--invite", SENTINEL], server, browser_ok=False)
    assert result.exit_code == 0, result.output
    assert "Could not auto-open" in result.output


# ---------------------------------------------------------------------------
# CLI: origin mismatch
# ---------------------------------------------------------------------------


def test_login_invite_from_another_server_fails_before_polling(
    hand_off_released, patched_default_path: Path
):
    server = _Server(_system_info("0.76.0"))
    other = "https://other.example.org"
    result, _, poll, browser = _invoke(["--invite", f"{other}/join/{SENTINEL}"], server)
    assert result.exit_code != 0
    assert "different server" in result.output
    assert "--server" in result.output
    assert other in result.output
    assert SENTINEL not in result.output
    poll.assert_not_called()
    browser.assert_not_called()
    assert not patched_default_path.exists()


def test_login_invite_scheme_mismatch_is_a_different_server(patched_default_path: Path):
    device = _device("http://localhost:3000/device", "http://localhost:3000/device?user_code=X")
    result, _, poll, _ = _invoke(
        ["--invite", f"https://localhost:3000/join/{SENTINEL}"], _Server(), device=device
    )
    assert result.exit_code != 0
    assert "different server" in result.output
    poll.assert_not_called()


def test_login_invite_under_base_path_on_same_host_is_accepted(patched_default_path: Path):
    result, _, poll, _ = _invoke(["--invite", f"{WEB}/app/join/{SENTINEL}"], _Server())
    assert result.exit_code == 0, result.output
    poll.assert_awaited_once()


# ---------------------------------------------------------------------------
# CLI: two-step fallback and beta_invites=false
# ---------------------------------------------------------------------------


def _assert_two_step(result, browser) -> None:
    assert result.exit_code == 0, result.output
    output = result.output
    assert "Accept your invite before you approve the code, in this order:" in output
    step1 = next(line for line in output.splitlines() if "1. " in line)
    step2 = next(line for line in output.splitlines() if "2. " in line)
    assert step1.rstrip().endswith(f"{WEB}/join/{SENTINEL}")
    assert step2.rstrip().endswith(DEVICE_URI_COMPLETE)
    assert "return_to" not in output
    assert "until the code expires (in 10 min)" in output
    # The browser opens only step 1.
    browser.assert_called_once_with(f"{WEB}/join/{SENTINEL}")


def test_login_invite_defaults_to_two_step_while_hand_off_unreleased(
    patched_default_path: Path,
):
    server = _Server(_system_info("99.0.0"))
    result, _, _, browser = _invoke(["--invite", SENTINEL], server)
    _assert_two_step(result, browser)


def test_login_invite_older_server_version_falls_back(
    hand_off_released, patched_default_path: Path
):
    server = _Server(_system_info("0.75.0"))
    result, _, _, browser = _invoke(["--invite", f"{WEB}/join/{SENTINEL}"], server)
    _assert_two_step(result, browser)


def test_login_invite_unparseable_version_falls_back(hand_off_released, patched_default_path: Path):
    server = _Server(_system_info("main-abc123"))
    result, _, _, browser = _invoke(["--invite", SENTINEL], server)
    _assert_two_step(result, browser)


def test_login_invite_system_info_404_falls_back(hand_off_released, patched_default_path: Path):
    server = _Server(status=404)
    result, _, _, browser = _invoke(["--invite", SENTINEL], server)
    _assert_two_step(result, browser)
    assert [r.url.path for r in server.requests] == ["/api/v1/system/info"]


def test_login_invite_system_info_timeout_falls_back(hand_off_released, patched_default_path: Path):
    server = _Server(exc=lambda r: httpx.ReadTimeout("timed out", request=r))
    result, _, _, browser = _invoke(["--invite", SENTINEL], server)
    _assert_two_step(result, browser)


def test_login_invite_expiry_comes_from_expires_in(patched_default_path: Path):
    result, _, _, _ = _invoke(["--invite", SENTINEL], _Server(), device=_device(expires_in=900))
    assert result.exit_code == 0, result.output
    assert "(in 15 min)" in result.output


def test_login_invite_two_step_no_browser(patched_default_path: Path):
    result, _, _, browser = _invoke(["--invite", SENTINEL, "--no-browser"], _Server())
    assert result.exit_code == 0, result.output
    browser.assert_not_called()
    assert f"{WEB}/join/{SENTINEL}" in result.output
    assert "--no-browser" in result.output


def test_login_invite_verification_uri_without_device_falls_back_to_given_link(
    hand_off_released, patched_default_path: Path
):
    device = _device(f"{WEB}/activate", f"{WEB}/activate?user_code=ABCD-1234")
    server = _Server(_system_info("0.76.0"))
    given = f"{WEB}/app/join/{SENTINEL}"
    result, _, _, browser = _invoke(["--invite", f"{given}?utm=1"], server, device=device)
    assert result.exit_code == 0, result.output
    assert "return_to" not in result.output
    # The server has the hand-off; only the URI shape stopped it, so the
    # lead-in must not claim the server cannot carry the invite.
    assert "Accept your invite before you approve the code" in result.output
    assert "cannot carry" not in result.output
    step1 = next(line for line in result.output.splitlines() if "1. " in line)
    step2 = next(line for line in result.output.splitlines() if "2. " in line)
    # /join cannot be placed next to /activate, so step 1 is the user's own link.
    assert step1.rstrip().endswith(given)
    assert step2.rstrip().endswith(f"{WEB}/activate?user_code=ABCD-1234")
    browser.assert_called_once_with(given)


def test_login_invite_verification_uri_without_device_and_bare_token(
    hand_off_released, patched_default_path: Path
):
    device = _device(f"{WEB}/activate", f"{WEB}/activate?user_code=ABCD-1234")
    server = _Server(_system_info("0.76.0"))
    result, _, poll, browser = _invoke(["--invite", SENTINEL], server, device=device)
    assert result.exit_code == 0, result.output
    # No link can be built: the CLI never invents a host, so it points the
    # user at the link they were sent and opens nothing.
    assert "invite link you were sent" in result.output
    assert SENTINEL not in result.output
    assert f"{WEB}/activate?user_code=ABCD-1234" in result.output
    browser.assert_not_called()
    poll.assert_awaited_once()


def test_login_invite_beta_invites_false_prints_normal_prompt(
    hand_off_released, patched_default_path: Path
):
    server = _Server(_system_info("0.76.0", beta_invites=False))
    result, _, poll, browser = _invoke(["--invite", SENTINEL], server)
    assert result.exit_code == 0, result.output
    assert "this server does not accept invites" in result.output
    assert "/join/" not in result.output
    assert SENTINEL not in result.output
    assert DEVICE_URI_COMPLETE in result.output
    browser.assert_called_once_with(DEVICE_URI_COMPLETE)
    poll.assert_awaited_once()


def test_login_invite_server_without_beta_invites_flag_prints_normal_prompt(
    patched_default_path: Path,
):
    """A server older than the flag (v0.17.1-v0.69.x) has no /join route."""
    server = _Server(_system_info("0.69.0", beta_invites=None))
    result, _, poll, browser = _invoke(["--invite", SENTINEL], server)
    assert result.exit_code == 0, result.output
    assert "this server does not accept invites" in result.output
    assert "/join/" not in result.output
    assert SENTINEL not in result.output
    browser.assert_called_once_with(DEVICE_URI_COMPLETE)
    poll.assert_awaited_once()


# ---------------------------------------------------------------------------
# CLI: the token never leaks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("released", [True, False], ids=["single-link", "two-step"])
def test_login_invite_token_only_appears_in_printed_links(
    released: bool, monkeypatch, caplog, patched_default_path: Path
):
    if released:
        monkeypatch.setattr(
            "kagura_memory.auth.device_flow.JOIN_RETURN_TO_MIN_SERVER_VERSION", HAND_OFF_VERSION
        )
    caplog.set_level(logging.DEBUG)
    server = _Server(_system_info("0.76.0"))
    result, authorize, poll, _ = _invoke(["--invite", f"{WEB}/join/{SENTINEL}"], server)
    assert result.exit_code == 0, result.output

    # Never sent to the server: not to device/authorize, the token poll or the probe.
    assert SENTINEL not in repr(authorize.call_args)
    assert SENTINEL not in repr(poll.call_args)
    token_lines = [line for line in result.output.splitlines() if SENTINEL in line]
    assert token_lines
    for line in token_lines:
        assert f"{WEB}/join/{SENTINEL}" in line
    assert SENTINEL not in caplog.text
    assert SENTINEL not in patched_default_path.read_text()
    for request in server.requests:
        assert SENTINEL not in str(request.url)
        assert SENTINEL not in request.content.decode()


def test_login_invite_token_not_in_poll_failure_message(patched_default_path: Path):
    poll = AsyncMock(
        side_effect=KaguraAuthExpiredError(
            "Device code expired before user approval. Run: kagura auth login"
        )
    )
    result, _, _, _ = _invoke(["--invite", SENTINEL, "--no-browser"], _Server(), poll=poll)
    assert result.exit_code != 0
    error = result.output[result.output.index("Error:") :]
    assert "Device code expired" in error
    assert SENTINEL not in error
    assert not patched_default_path.exists()
