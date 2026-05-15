"""Tests for `kagura resource ...` CLI commands.

Focused on the post-#117 credential resolution chain: `_get_resource_client`
now walks `_resolve_auth` (env > OAuth profile > .kagura.json) instead of
short-circuiting on `.kagura.json` only. Matches the Files CLI matrix
(post-#118) so an OAuth-only operator can run `kagura resource ...`
end-to-end, except for `kagura resource setup` which is documented as
static-api_key-only in #117.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kagura_memory.auth.credentials import (
    CredentialsFile,
    OAuthCredentials,
    reset_state_cache,
    save_credentials_file,
)
from kagura_memory.cli import main


@pytest.fixture(autouse=True)
def _isolate_oauth_state(tmp_path, monkeypatch):
    """Isolate every test from real ``~/.kagura/credentials.json`` and env.

    Mirrors the autouse isolation added in ``test_cli_files.py`` for the
    same reason: the post-#117 resolver consults the OAuth profile before
    ``.kagura.json``, so a developer's stored profile would otherwise
    pre-empt config-only fixtures.
    """
    fake_path = tmp_path / "default-credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    reset_state_cache()
    yield
    reset_state_cache()


def _make_oauth_creds(
    workspace_id: str = "00000000-0000-0000-0000-0000000000ff",
    expires_in_seconds: int = 3600,
) -> OAuthCredentials:
    return OAuthCredentials(
        server="https://memory.kagura-ai.com",
        mcp_url="https://memory.kagura-ai.com/mcp",
        client_id="kagura-cli",
        access_token="atok-cli-resource-test",
        refresh_token="rtok-cli-resource-test",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
        scope="memory:read memory:write",
        workspace_id=workspace_id,
        workspace_name="cli-resource-test",
        user_email="resource@example.com",
        issued_at=datetime.now(UTC),
    )


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
    # from_mcp_url is the entry point — assert the resolver chain ran.
    mock_cls.from_mcp_url.assert_called_once()
    kwargs = mock_cls.from_mcp_url.call_args.kwargs
    assert kwargs["api_key"] is None  # always None — the chain resolves
    assert kwargs["mcp_url"] is None  # config had none


def test_resource_list_uses_oauth_profile():
    """OAuth profile on disk + no env + no config → ResourceClient built via OAuth path."""
    # The autouse fixture redirected DEFAULT_CREDENTIALS_PATH; write the
    # profile there so the resolver finds it during priority-3.
    import kagura_memory.auth.credentials as _creds

    cf = CredentialsFile()
    cf.set_profile("default", _make_oauth_creds())
    save_credentials_file(cf, _creds.DEFAULT_CREDENTIALS_PATH)
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
    mock_cls.from_mcp_url.assert_called_once()


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
    # from_mcp_url is called with api_key=None — the chain finds the
    # config fallback internally via _resolve_auth priority 4.
    kwargs = mock_cls.from_mcp_url.call_args.kwargs
    assert kwargs["api_key"] is None
    assert kwargs["mcp_url"] == "https://test.com/mcp"


def test_resource_list_no_credentials_errors():
    """No env, no OAuth, no config api_key → resolver raises → ClickException surfaces."""
    with patch("kagura_memory.cli.load_config", return_value={"api_key": ""}):
        # Also short-circuit the priority-4 reload inside _resolve_auth.
        with patch("kagura_memory._auth.load_config", return_value={"api_key": ""}):
            runner = CliRunner()
            result = runner.invoke(main, ["resource", "list"])

    assert result.exit_code != 0, result.output
    assert "No credentials" in result.output or "kagura auth login" in result.output


def test_resource_setup_in_oauth_mode_errors():
    """OAuth-mode ``kagura resource setup`` surfaces the NotImplementedError as a clean CLI error.

    ``ResourceClient.setup_resource`` raises ``NotImplementedError`` in
    OAuth mode (intentionally out of scope for #117). ``_run_resource_command``
    wraps unexpected exceptions in ``ClickException`` so the operator
    sees a clean error message and an actionable hint instead of a
    traceback.
    """
    import kagura_memory.auth.credentials as _creds

    cf = CredentialsFile()
    cf.set_profile("default", _make_oauth_creds())
    save_credentials_file(cf, _creds.DEFAULT_CREDENTIALS_PATH)
    reset_state_cache()

    with patch("kagura_memory.cli.load_config", return_value={}):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["resource", "setup", "--resource-id", "demo"],
        )

    assert result.exit_code != 0, result.output
    assert "OAuth mode" in result.output
    assert "static api_key" in result.output or "KAGURA_API_KEY" in result.output


def test_resource_list_env_wins_over_config_api_key(monkeypatch):
    """KAGURA_API_KEY env + ``.kagura.json`` api_key → env wins (mirrors #118 Files CLI).

    The resolver chain is canonical SDK order: ``env > OAuth > config``.
    ``_get_resource_client`` walks the same chain, so a session with
    both env and config api_key picks env — same precedence as
    ``kagura context list`` and (post-#118) ``kagura files list``.
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
    # The chain is opaque from the CLI's perspective — we cannot directly
    # assert env wins without inspecting _resolve_auth internals. But we
    # can verify the CLI passed api_key=None so the chain (not the CLI)
    # decides. The env-wins behavior is locked in by test_kagura_api_key_
    # env_wins in tests/test_client_auth_resolution.py.
    kwargs = mock_cls.from_mcp_url.call_args.kwargs
    assert kwargs["api_key"] is None
