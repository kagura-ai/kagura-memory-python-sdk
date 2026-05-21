"""Tests for `kagura resource ...` CLI commands.

`_get_resource_client` walks `_resolve_auth` (env > OAuth profile >
.kagura.json), matching `KaguraClient` and `FilesClient`. Tests cover
the four resolver branches plus the OAuth-mode `setup` guard.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kagura_memory.auth.credentials import (
    CredentialsFile,
    reset_state_cache,
    save_credentials_file,
)
from kagura_memory.cli import main
from kagura_memory.resource_client import _SETUP_OAUTH_NOT_SUPPORTED_MSG
from tests.conftest import make_oauth_creds


@pytest.fixture(autouse=True)
def _isolate_oauth_state(tmp_path, monkeypatch):
    """Isolate every test from real ``~/.kagura/credentials.json`` and env.

    The post-issue-#117 resolver consults the OAuth profile before
    ``.kagura.json``, so a developer's stored profile would otherwise
    pre-empt config-only fixtures. Yields the redirected credentials path
    so tests that need to write an OAuth profile can target it without
    re-importing the patched module attribute.
    """
    fake_path = tmp_path / "default-credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    reset_state_cache()
    yield fake_path
    reset_state_cache()


def _mock_resource_client() -> MagicMock:
    """Build an async-context-manager-shaped ResourceClient mock."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _wire_resource_client_mock(mock_cls: MagicMock, mock_client: MagicMock) -> None:
    """Route every CLI construction path on the patched ResourceClient to mock_client."""
    mock_cls.return_value = mock_client
    mock_cls._from_resolved_auth.return_value = mock_client
    mock_cls.from_mcp_url.return_value = mock_client


def _resolved_from_call(mock_cls: MagicMock):
    """Pull the resolved auth dataclass out of the ``_from_resolved_auth`` call.

    ``_get_resource_client`` constructs via ``ResourceClient._from_resolved_auth``
    directly, so its first positional argument is the ``_StaticAuth | _OAuthAuth``
    dataclass produced by ``_resolve_auth``. Tests inspect this to verify which
    resolver branch fired.
    """
    return mock_cls._from_resolved_auth.call_args.args[0]


def test_resource_list_uses_env_api_key(monkeypatch):
    """KAGURA_API_KEY env → ResourceClient built with the env api_key via SDK chain."""
    monkeypatch.setenv("KAGURA_API_KEY", "env-key")

    mock_client = _mock_resource_client()
    mock_client.list_resources.return_value = MagicMock(model_dump_json=lambda **_: "{}")

    with (
        patch("kagura_memory.cli.load_config", return_value={}),
        patch("kagura_memory.cli.ResourceClient") as mock_cls,
    ):
        _wire_resource_client_mock(mock_cls, mock_client)
        runner = CliRunner()
        result = runner.invoke(main, ["resource", "list"])

    assert result.exit_code == 0, result.output
    resolved = _resolved_from_call(mock_cls)
    assert resolved.source == "env"
    assert resolved.api_key == "env-key"


def test_resource_list_uses_oauth_profile(_isolate_oauth_state):
    """OAuth profile on disk + no env + no config → ResourceClient built via OAuth path.

    Pins that api_key + workspace_id + mcp_url all come from the OAuth
    profile (the same-source pairing invariant from #115). The CLI
    passes ``mcp_url=None`` to the resolver so each priority branch
    pairs its credential with its own URL source — for the OAuth
    branch that's the profile's stored ``mcp_url``.
    """
    from kagura_memory._auth import _OAuthAuth

    cf = CredentialsFile()
    cf.set_profile("default", make_oauth_creds())
    save_credentials_file(cf, _isolate_oauth_state)
    reset_state_cache()

    mock_client = _mock_resource_client()
    mock_client.list_resources.return_value = MagicMock(model_dump_json=lambda **_: "{}")

    with (
        patch("kagura_memory.cli.load_config", return_value={}),
        patch("kagura_memory.cli.ResourceClient") as mock_cls,
    ):
        _wire_resource_client_mock(mock_cls, mock_client)
        runner = CliRunner()
        result = runner.invoke(main, ["resource", "list"])

    assert result.exit_code == 0, result.output
    resolved = _resolved_from_call(mock_cls)
    assert isinstance(resolved, _OAuthAuth)


def test_resource_list_falls_back_to_config_api_key():
    """No env, no OAuth, only ``.kagura.json`` → resolver priority-4 fires."""
    mock_client = _mock_resource_client()
    mock_client.list_resources.return_value = MagicMock(model_dump_json=lambda **_: "{}")

    config_dict = {"api_key": "config-key", "mcp_url": "https://test.com/mcp"}
    with (
        patch("kagura_memory.cli.load_config", return_value=config_dict),
        patch("kagura_memory.cli.ResourceClient") as mock_cls,
    ):
        _wire_resource_client_mock(mock_cls, mock_client)
        runner = CliRunner()
        result = runner.invoke(main, ["resource", "list"])

    assert result.exit_code == 0, result.output
    resolved = _resolved_from_call(mock_cls)
    assert resolved.source == "config"
    assert resolved.api_key == "config-key"


def test_resource_list_no_credentials_errors():
    """No env, no OAuth, no config api_key → resolver raises → ClickException surfaces."""
    with patch("kagura_memory.cli.load_config", return_value={"api_key": ""}):
        # Also short-circuit the priority-4 reload inside _resolve_auth.
        with patch("kagura_memory._auth.load_config", return_value={"api_key": ""}):
            runner = CliRunner()
            result = runner.invoke(main, ["resource", "list"])

    assert result.exit_code != 0, result.output
    assert "No credentials" in result.output or "kagura auth login" in result.output


def test_resource_setup_in_oauth_mode_errors(_isolate_oauth_state):
    """OAuth-mode ``kagura resource setup`` surfaces the NotImplementedError as a clean CLI error.

    ``ResourceClient.setup_resource`` raises ``NotImplementedError`` in
    OAuth mode (the underlying ``KaguraClient`` MCP call does not yet
    accept an OAuth httpx.Auth). ``_run_resource_command`` wraps
    unexpected exceptions in ``ClickException`` so the operator sees a
    clean error message and an actionable hint instead of a traceback.
    """
    cf = CredentialsFile()
    cf.set_profile("default", make_oauth_creds())
    save_credentials_file(cf, _isolate_oauth_state)
    reset_state_cache()

    with patch("kagura_memory.cli.load_config", return_value={}):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["resource", "setup", "--resource-id", "demo"],
        )

    assert result.exit_code != 0, result.output
    # Assert against the canonical message constant rather than substrings —
    # ``_run_resource_command`` wraps via ``_exc_message(e)`` which returns
    # ``str(e)`` when non-empty, so the full sentence reaches the operator
    # for any wrapper whose inner exception carries a message (the empty-
    # ``str(e)`` fallback to class name is tested separately in
    # ``tests/test_exceptions.py``).
    assert _SETUP_OAUTH_NOT_SUPPORTED_MSG in result.output


def test_resource_list_oauth_profile_mcp_url_not_overridden_by_config(_isolate_oauth_state):
    """OAuth profile's stored ``mcp_url`` must reach the wire when the
    OAuth branch resolves the credential, even when ``.kagura.json``
    (or its env-default fallback) supplies a different ``mcp_url``.

    Regression test for a bug surfaced in PR #119 review: the CLI used
    to forward ``config.get("mcp_url") or None`` as the explicit
    ``mcp_url`` argument to ``_resolve_auth``. Because ``load_config()``
    returns the default cloud URL when ``.kagura.json`` is absent, the
    override path fired on every OAuth-only invocation — routing
    OAuth users bound to a non-default server to the default cloud
    host. The fix passes ``mcp_url=None`` so each resolver branch
    pairs its credential with its own URL source.
    """
    custom_url = "https://custom.example.com/mcp"
    cf = CredentialsFile()
    cf.set_profile(
        "default",
        make_oauth_creds(server="https://custom.example.com"),
    )
    save_credentials_file(cf, _isolate_oauth_state)
    reset_state_cache()

    mock_client = _mock_resource_client()
    mock_client.list_resources.return_value = MagicMock(model_dump_json=lambda **_: "{}")

    # Config supplies a DIFFERENT mcp_url to ensure the OAuth branch
    # wins by URL — not just by api_key.
    with (
        patch(
            "kagura_memory.cli.load_config",
            return_value={"mcp_url": "https://stale-config.example.com/mcp"},
        ),
        patch("kagura_memory.cli.ResourceClient") as mock_cls,
    ):
        _wire_resource_client_mock(mock_cls, mock_client)
        runner = CliRunner()
        result = runner.invoke(main, ["resource", "list"])

    assert result.exit_code == 0, result.output
    resolved = _resolved_from_call(mock_cls)
    assert resolved.mcp_url == custom_url, (
        f"OAuth profile's stored mcp_url should win, got {resolved.mcp_url}"
    )


def test_resource_list_env_wins_over_config_api_key(monkeypatch):
    """KAGURA_API_KEY env + ``.kagura.json`` api_key → env wins.

    The resolver chain is canonical SDK order: ``env > OAuth > config``.
    ``_get_resource_client`` walks the same chain, so a session with
    both env and config api_key picks env — locked in here against
    future drift.
    """
    monkeypatch.setenv("KAGURA_API_KEY", "env-key")

    mock_client = _mock_resource_client()
    mock_client.list_resources.return_value = MagicMock(model_dump_json=lambda **_: "{}")

    with (
        patch(
            "kagura_memory.cli.load_config",
            return_value={"api_key": "config-key", "mcp_url": "https://test.com/mcp"},
        ),
        patch("kagura_memory.cli.ResourceClient") as mock_cls,
    ):
        _wire_resource_client_mock(mock_cls, mock_client)
        runner = CliRunner()
        result = runner.invoke(main, ["resource", "list"])

    assert result.exit_code == 0, result.output
    resolved = _resolved_from_call(mock_cls)
    assert resolved.source == "env"
    assert resolved.api_key == "env-key"
