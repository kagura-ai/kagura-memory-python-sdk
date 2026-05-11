"""Tests for `kagura files ...` CLI commands."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

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
def test_files_upload_missing_api_key(mock_config, tmp_path):
    """upload with context-id provided but no api_key → No API key error."""
    mock_config.return_value = {"api_key": "", "context_id": SAMPLE_CTX_ID}
    p = tmp_path / "hello.txt"
    p.write_text("hi")
    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p)])
    assert result.exit_code != 0
    assert "No API key" in result.output


@patch("kagura_memory.cli.load_config")
def test_files_upload_missing_context(mock_config, tmp_path):
    """upload without --context-id and without .kagura.json default → error."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
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
