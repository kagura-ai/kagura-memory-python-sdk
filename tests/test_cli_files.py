"""Tests for `kagura files ...` CLI commands."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from kagura_memory.auth.credentials import (
    CredentialsFile,
    OAuthCredentials,
    reset_state_cache,
    save_credentials_file,
)
from kagura_memory.cli import main
from kagura_memory.models import FileListResponse, FileObject

SAMPLE_CTX_ID = "00000000-0000-0000-0000-000000000001"
SAMPLE_FILE_ID = "10000000-0000-0000-0000-000000000002"


def _file_object() -> FileObject:
    return FileObject(
        id=SAMPLE_FILE_ID,
        workspace_id=SAMPLE_CTX_ID,
        filename="hello.txt",
        content_type="text/plain",
        size_bytes=18,
        sha256="a" * 64,
        status="uploaded",
        created_at=datetime(2026, 5, 11, tzinfo=UTC),
        uploaded_at=datetime(2026, 5, 11, 0, 0, 1, tzinfo=UTC),
    )


def _mock_files_client(method_name: str, return_value) -> MagicMock:
    """Build a FilesClient mock with one method stubbed."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    getattr(client, method_name).return_value = return_value
    return client


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.FilesClient")
def test_files_upload(mock_client_cls, mock_config, tmp_path):
    """`kagura files upload` calls FilesClient.upload and prints the FileObject."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_client = _mock_files_client("upload", _file_object())
    # FilesClient.from_mcp_url is a classmethod — patch it too
    mock_client_cls.from_mcp_url.return_value = mock_client

    p = tmp_path / "hello.txt"
    p.write_text("hello kagura files")

    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p)])

    assert result.exit_code == 0, result.output
    assert SAMPLE_FILE_ID in result.output
    assert "uploaded" in result.output

    mock_client.upload.assert_awaited_once()
    call = mock_client.upload.call_args
    assert call.kwargs["context_id"] == SAMPLE_CTX_ID
    assert call.kwargs["source"] == p


@patch("kagura_memory.cli.load_config")
def test_files_upload_missing_credentials(mock_config, monkeypatch, tmp_path):
    """upload with context-id but no api_key + no OAuth profile → credentials error."""
    mock_config.return_value = {"api_key": "", "context_id": SAMPLE_CTX_ID}
    # Block every credential source so the resolver chain reaches its terminal raise.
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    monkeypatch.setattr("kagura_memory._auth.load_config", lambda: {"api_key": ""})
    p = tmp_path / "hello.txt"
    p.write_text("hi")
    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p)])
    assert result.exit_code != 0
    assert "No credentials found" in result.output or "kagura auth login" in result.output


@patch("kagura_memory.cli.load_config")
def test_files_upload_missing_context(mock_config, monkeypatch, tmp_path):
    """upload without --context-id and without any workspace source → context_id required."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    # Block the OAuth profile workspace_id fallback so context_id stays empty.
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    p = tmp_path / "hello.txt"
    p.write_text("hi")
    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p)])
    assert result.exit_code != 0
    assert "context_id required" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.FilesClient")
def test_files_download_url(mock_client_cls, mock_config):
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_client = _mock_files_client("download_url", "https://r2.example.com/get/key?sig=...")
    mock_client_cls.from_mcp_url.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["files", "download-url", SAMPLE_FILE_ID])

    assert result.exit_code == 0, result.output
    assert "https://r2.example.com/get/key" in result.output
    mock_client.download_url.assert_awaited_once_with(SAMPLE_FILE_ID)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.FilesClient")
def test_files_delete(mock_client_cls, mock_config):
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_client = _mock_files_client("delete", None)
    mock_client_cls.from_mcp_url.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["files", "delete", SAMPLE_FILE_ID])

    assert result.exit_code == 0, result.output
    assert f"Deleted {SAMPLE_FILE_ID}" in result.output
    mock_client.delete.assert_awaited_once_with(SAMPLE_FILE_ID)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.FilesClient")
def test_files_unexpected_exception_wrapped_as_click(mock_client_cls, mock_config, tmp_path):
    """Non-ClickException raised inside the runner → wrapped as 'Error: ...'."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_client_cls.from_mcp_url.side_effect = RuntimeError("boom from factory")

    p = tmp_path / "hello.txt"
    p.write_text("hi")
    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p)])

    assert result.exit_code != 0
    assert "Error: boom from factory" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.FilesClient")
def test_files_list(mock_client_cls, mock_config):
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_client = _mock_files_client(
        "list",
        FileListResponse(files=[_file_object()], next_cursor=None),
    )
    mock_client_cls.from_mcp_url.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["files", "list"])

    assert result.exit_code == 0, result.output
    assert SAMPLE_FILE_ID in result.output

    call = mock_client.list.call_args
    assert call.kwargs["context_id"] == SAMPLE_CTX_ID
    assert call.kwargs["limit"] == 50


# ============================================================================
# OAuth credentials.json-only flow (#110 headline scenario)
# ============================================================================


def _oauth_creds_with_workspace(workspace_id: str) -> OAuthCredentials:
    return OAuthCredentials(
        server="https://oauth.example.com",
        mcp_url="https://oauth.example.com/mcp",
        client_id="kagura-cli",
        access_token="atok-cli-test",
        refresh_token="rtok-cli-test",
        token_type="Bearer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        scope="memory:write",
        workspace_id=workspace_id,
        workspace_name="cli-test-ws",
        user_email="cli@example.com",
        issued_at=datetime.now(UTC),
    )


def test_files_upload_uses_oauth_workspace_id_when_no_context(monkeypatch, tmp_path):
    """`.kagura.json` absent + credentials.json present → workspace_id from OAuth profile.

    Headline scenario from #110: ``kagura auth login`` populates the
    profile's ``workspace_id``; subsequent ``kagura files upload`` with
    neither ``--context-id`` nor ``.kagura.json.context_id`` must resolve
    the workspace via the OAuth profile and succeed.
    """
    fake_creds_path = tmp_path / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_creds_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)

    cf = CredentialsFile()
    cf.set_profile("default", _oauth_creds_with_workspace(SAMPLE_CTX_ID))
    save_credentials_file(cf, fake_creds_path)
    reset_state_cache()

    with (
        patch("kagura_memory.cli.load_config", return_value={}),
        patch("kagura_memory.cli.FilesClient") as mock_client_cls,
    ):
        mock_client = _mock_files_client("upload", _file_object())
        mock_client_cls.from_mcp_url.return_value = mock_client

        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p)])

    reset_state_cache()

    assert result.exit_code == 0, result.output
    mock_client.upload.assert_awaited_once()
    assert mock_client.upload.call_args.kwargs["context_id"] == SAMPLE_CTX_ID


def test_files_upload_treats_context_id_auto_as_unset(monkeypatch, tmp_path):
    """`.kagura.json` with ``context_id: "auto"`` should not reach the SDK.

    The CLI converts the ``"auto"`` sentinel to the OAuth profile's
    workspace_id before calling ``FilesClient.upload``. Previously this
    would forward ``"auto"`` straight to the server and hit a 422
    ``workspace_id`` validation error.
    """
    fake_creds_path = tmp_path / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_creds_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)

    cf = CredentialsFile()
    cf.set_profile("default", _oauth_creds_with_workspace(SAMPLE_CTX_ID))
    save_credentials_file(cf, fake_creds_path)
    reset_state_cache()

    from kagura_memory.cli import _CONTEXT_ID_AUTO

    with (
        patch(
            "kagura_memory.cli.load_config",
            return_value={"api_key": "key", "context_id": _CONTEXT_ID_AUTO},
        ),
        patch("kagura_memory.cli.FilesClient") as mock_client_cls,
    ):
        mock_client = _mock_files_client("upload", _file_object())
        mock_client_cls.from_mcp_url.return_value = mock_client

        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p)])

    reset_state_cache()

    assert result.exit_code == 0, result.output
    assert mock_client.upload.call_args.kwargs["context_id"] == SAMPLE_CTX_ID
    assert mock_client.upload.call_args.kwargs["context_id"] != _CONTEXT_ID_AUTO
