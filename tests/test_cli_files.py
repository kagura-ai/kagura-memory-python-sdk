"""Tests for `kagura files ...` CLI commands."""

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
from kagura_memory.models import FileListResponse, FileObject


@pytest.fixture(autouse=True)
def _isolate_oauth_state(tmp_path, monkeypatch):
    """Isolate every test from real ``~/.kagura/credentials.json`` and env.

    Post-issue #118 the Files CLI walks the canonical SDK chain
    (``env > OAuth profile > .kagura.json``), so any OAuth profile a
    developer has stored on their machine would otherwise pre-empt the
    config-only test fixtures and silently change behavior. Each test
    starts from a clean credentials path and clean env; tests that need
    OAuth state set it up explicitly.
    """
    fake_path = tmp_path / "default-credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    reset_state_cache()
    yield
    reset_state_cache()


SAMPLE_CTX_ID = "00000000-0000-0000-0000-000000000001"
SAMPLE_FILE_ID = "10000000-0000-0000-0000-000000000002"
SAMPLE_BINDING_CTX_ID = "00000000-0000-0000-0000-0000000000bb"


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


def _wire_files_client_mock(mock_client_cls: MagicMock, mock_client: MagicMock) -> None:
    """Route every CLI entry point on the patched FilesClient to ``mock_client``.

    The CLI may construct via ``FilesClient(...)`` (direct) or
    ``FilesClient._from_resolved_auth(...)`` (the internal classmethod
    shared with ``from_mcp_url``). Configure all entry points so test
    setup stays insensitive to which one the CLI happens to pick.
    """
    mock_client_cls.return_value = mock_client
    mock_client_cls._from_resolved_auth.return_value = mock_client
    mock_client_cls.from_mcp_url.return_value = mock_client


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
    # The CLI constructs FilesClient(...) directly via _build_files_client_from_auth;
    # set return_value so any call to the patched class returns the mock client.
    _wire_files_client_mock(mock_client_cls, mock_client)

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
@patch("kagura_memory.cli.FilesClient")
def test_files_upload_forwards_binding_context_id(mock_client_cls, mock_config, tmp_path):
    """`files upload --binding-context-id <ctx>` threads binding_context_id to upload()."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_client = _mock_files_client("upload", _file_object())
    _wire_files_client_mock(mock_client_cls, mock_client)

    p = tmp_path / "hello.txt"
    p.write_text("hi")
    runner = CliRunner()
    result = runner.invoke(
        main, ["files", "upload", str(p), "--binding-context-id", SAMPLE_BINDING_CTX_ID]
    )

    assert result.exit_code == 0, result.output
    assert mock_client.upload.call_args.kwargs["binding_context_id"] == SAMPLE_BINDING_CTX_ID


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.FilesClient")
def test_files_upload_without_binding_flag_passes_none(mock_client_cls, mock_config, tmp_path):
    """Without --binding-context-id, the CLI passes binding_context_id=None (NULL-context)."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_client = _mock_files_client("upload", _file_object())
    _wire_files_client_mock(mock_client_cls, mock_client)

    p = tmp_path / "hello.txt"
    p.write_text("hi")
    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p)])

    assert result.exit_code == 0, result.output
    assert mock_client.upload.call_args.kwargs["binding_context_id"] is None


def _mock_kagura_client(mock_client_cls: MagicMock) -> MagicMock:
    """Wire a KaguraClient mock whose ``remember`` returns a memory id."""
    client = AsyncMock()
    client.remember.return_value = {"memory_id": "mem-1"}
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = client
    return client


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
@patch("kagura_memory.cli.FilesClient")
def test_files_upload_remember_creates_linked_memory(
    mock_files_cls, mock_kagura_cls, mock_config, tmp_path
):
    """`files upload --remember` uploads, then creates a memory linked to the file_object."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_files = _mock_files_client("upload", _file_object())
    _wire_files_client_mock(mock_files_cls, mock_files)
    mock_kagura = _mock_kagura_client(mock_kagura_cls)

    p = tmp_path / "hello.txt"
    p.write_text("hello kagura files")

    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p), "--remember"])

    assert result.exit_code == 0, result.output
    # Output reports both the file_object and the created memory.
    assert SAMPLE_FILE_ID in result.output
    assert "mem-1" in result.output

    mock_kagura.remember.assert_awaited_once()
    kwargs = mock_kagura.remember.call_args.kwargs
    assert kwargs["context_id"] == SAMPLE_CTX_ID
    assert kwargs["source_type"] == "file"
    assert kwargs["source_uri"] == p.resolve().as_uri()
    # The memory links back to the file_object via details.file_id.
    assert kwargs["details"]["file_id"] == SAMPLE_FILE_ID
    # The memory client must self-resolve credentials (constructed with no
    # api_key/mcp_url args) so it shares the exact source that uploaded the
    # file and owns ``ctx``. Passing config values would force the "explicit"
    # branch of _resolve_auth and diverge from the upload's env/OAuth source
    # for multi-source users → cross-workspace 403.
    mock_kagura_cls.assert_called_once_with()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
@patch("kagura_memory.cli.FilesClient")
def test_files_upload_without_remember_skips_memory(
    mock_files_cls, mock_kagura_cls, mock_config, tmp_path
):
    """Without --remember, no memory is created (default path unchanged)."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_files = _mock_files_client("upload", _file_object())
    _wire_files_client_mock(mock_files_cls, mock_files)
    mock_kagura = _mock_kagura_client(mock_kagura_cls)

    p = tmp_path / "hello.txt"
    p.write_text("hi")

    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p)])

    assert result.exit_code == 0, result.output
    mock_kagura.remember.assert_not_awaited()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
@patch("kagura_memory.cli.FilesClient")
def test_files_upload_remember_custom_summary_type_tags(
    mock_files_cls, mock_kagura_cls, mock_config, tmp_path
):
    """--remember forwards --summary / --type / --importance / --tags to remember()."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_files = _mock_files_client("upload", _file_object())
    _wire_files_client_mock(mock_files_cls, mock_files)
    mock_kagura = _mock_kagura_client(mock_kagura_cls)

    p = tmp_path / "hello.txt"
    p.write_text("hi")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "files",
            "upload",
            str(p),
            "--remember",
            "--summary",
            "My doc",
            "--type",
            "doc",
            "--importance",
            "0.9",
            "--tags",
            "a, b",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = mock_kagura.remember.call_args.kwargs
    assert kwargs["summary"] == "My doc"
    assert kwargs["type"] == "doc"
    assert kwargs["importance"] == 0.9
    assert kwargs["tags"] == ["a", "b"]


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
@patch("kagura_memory.cli.FilesClient")
def test_files_upload_remember_failure_still_reports_file_id(
    mock_files_cls, mock_kagura_cls, mock_config, tmp_path
):
    """If upload succeeds but the memory write fails, the file_id must not be lost.

    Otherwise the user sees a bare error, does not learn the file_object was
    already created, and re-runs — orphaning a duplicate upload.
    """
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_files = _mock_files_client("upload", _file_object())
    _wire_files_client_mock(mock_files_cls, mock_files)
    mock_kagura = _mock_kagura_client(mock_kagura_cls)
    mock_kagura.remember.side_effect = RuntimeError("boom")

    p = tmp_path / "hello.txt"
    p.write_text("hi")

    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p), "--remember"])

    assert result.exit_code != 0
    # The successful upload's file_id is surfaced despite the memory failure.
    assert SAMPLE_FILE_ID in result.output


@pytest.mark.parametrize(
    "bad_result",
    [
        {"status": "error", "message": "quota exceeded"},  # MCP domain error
        {"ok": True},  # missing memory_id
    ],
)
@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
@patch("kagura_memory.cli.FilesClient")
def test_files_upload_remember_domain_error_is_failure(
    mock_files_cls, mock_kagura_cls, mock_config, bad_result, tmp_path
):
    """remember() returning an error-shaped dict (no raise) must NOT exit 0.

    KaguraClient.remember() surfaces MCP domain errors as a dict with
    status=="error" / no memory_id rather than raising; treating that as
    success would print an error payload as if the memory was created.
    Mirrors the ingest path (ingestor.py).
    """
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    mock_files = _mock_files_client("upload", _file_object())
    _wire_files_client_mock(mock_files_cls, mock_files)
    mock_kagura = _mock_kagura_client(mock_kagura_cls)
    mock_kagura.remember.return_value = bad_result

    p = tmp_path / "hello.txt"
    p.write_text("hi")

    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p), "--remember"])

    assert result.exit_code != 0
    # The upload succeeded — file_id still surfaced for recovery.
    assert SAMPLE_FILE_ID in result.output


@patch("kagura_memory.cli.load_config")
def test_files_upload_summary_without_remember_errors(mock_config, tmp_path):
    """--summary/--tags only make sense with --remember; using them alone must error,
    not silently drop the value."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": SAMPLE_CTX_ID,
    }
    p = tmp_path / "hello.txt"
    p.write_text("hi")

    runner = CliRunner()
    result = runner.invoke(main, ["files", "upload", str(p), "--summary", "x"])

    assert result.exit_code != 0
    assert "--remember" in result.output


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
    """upload with config api_key but no context_id and no -c → same-source error.

    After issue #115, falling through to the OAuth profile's
    workspace_id is no longer allowed when api_key comes from a
    different source. With api_key in ``.kagura.json`` and no
    ``context_id``, the CLI raises an actionable error pointing to
    ``.kagura.json``'s ``context_id`` field or ``--context-id``.
    """
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
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
    assert "context_id is missing" in result.output
    assert "--context-id" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.FilesClient")
def test_files_download_url(mock_client_cls, mock_config):
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    mock_client = _mock_files_client("download_url", "https://r2.example.com/get/key?sig=...")
    _wire_files_client_mock(mock_client_cls, mock_client)

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
    _wire_files_client_mock(mock_client_cls, mock_client)

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
    # The CLI builds via FilesClient._from_resolved_auth(); have it raise.
    mock_client_cls._from_resolved_auth.side_effect = RuntimeError("boom from factory")

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
    _wire_files_client_mock(mock_client_cls, mock_client)

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


def test_files_upload_strips_whitespace_from_explicit_context_id(monkeypatch, tmp_path):
    """``--context-id`` with surrounding whitespace is normalized before resolution.

    Defends against a class of subtle bugs where a whitespace-padded UUID
    (e.g. copy-paste from a chat client) would pass the truthy check but
    then either fail UUID validation downstream or, worse, pass through
    the ``"auto"`` sentinel comparison without matching the sentinel.
    """
    ws = "20000000-0000-0000-0000-000000000003"
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    reset_state_cache()

    with (
        patch(
            "kagura_memory.cli.load_config",
            return_value={"api_key": "key", "mcp_url": "https://test.com/mcp"},
        ),
        patch("kagura_memory.cli.FilesClient") as mock_client_cls,
    ):
        mock_client = _mock_files_client("upload", _file_object())
        _wire_files_client_mock(mock_client_cls, mock_client)

        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p), "--context-id", f"  {ws}  "])

    reset_state_cache()

    assert result.exit_code == 0, result.output
    # Whitespace was stripped before reaching FilesClient.upload.
    assert mock_client.upload.call_args.kwargs["context_id"] == ws


def test_files_upload_with_explicit_context_id_flag(monkeypatch, tmp_path):
    """``--context-id`` flag is the highest-priority source and bypasses every fallback.

    The first branch of :func:`_resolve_workspace_from_source`: an
    explicit UUID via ``-c`` / ``--context-id`` wins over the
    api_key's same-source workspace and over the OAuth profile.
    """
    explicit_ctx = "20000000-0000-0000-0000-000000000003"
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    reset_state_cache()

    with (
        patch(
            "kagura_memory.cli.load_config",
            return_value={
                "api_key": "key",
                "mcp_url": "https://test.com/mcp",
                "context_id": SAMPLE_CTX_ID,
            },
        ),
        patch("kagura_memory.cli.FilesClient") as mock_client_cls,
    ):
        mock_client = _mock_files_client("upload", _file_object())
        _wire_files_client_mock(mock_client_cls, mock_client)

        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p), "--context-id", explicit_ctx])

    reset_state_cache()

    assert result.exit_code == 0, result.output
    assert mock_client.upload.call_args.kwargs["context_id"] == explicit_ctx
    # Confirms the flag won over the .kagura.json default.
    assert mock_client.upload.call_args.kwargs["context_id"] != SAMPLE_CTX_ID


def test_files_upload_uses_oauth_workspace_id_when_no_context(monkeypatch, tmp_path):
    """`.kagura.json` absent + credentials.json present → workspace_id from OAuth profile.

    Headline scenario from #110: ``kagura auth login`` populates the
    profile's ``workspace_id``; subsequent ``kagura files upload`` with
    neither ``--context-id`` nor ``.kagura.json.context_id`` must
    resolve the workspace via the OAuth profile and succeed. With #115
    this is the OAuth same-source pair — api_key + workspace_id both
    come from the OAuth profile.
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
        _wire_files_client_mock(mock_client_cls, mock_client)

        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p)])

    reset_state_cache()

    assert result.exit_code == 0, result.output
    mock_client.upload.assert_awaited_once()
    assert mock_client.upload.call_args.kwargs["context_id"] == SAMPLE_CTX_ID


def test_files_upload_oauth_wins_over_config_when_both_present(monkeypatch, tmp_path):
    """``.kagura.json`` api_key + OAuth profile (no env, no -c) → OAuth wins entirely.

    Issue #118 aligned the Files CLI with the SDK chain
    ``env > OAuth > .kagura.json``. When both config api_key and an
    OAuth profile exist, OAuth wins entirely for the credential
    triple: ``api_key`` (Bearer via ``KaguraOAuth``), ``workspace_id``,
    and ``mcp_url`` all come from the same OAuth profile (the
    same-source invariant from #115, preserved structurally).

    The CLI passes ``mcp_url=None`` to the resolver so each priority
    branch pairs its credential with its own URL source — see
    ``test_cli_resource.py::test_resource_list_oauth_profile_mcp_url_not_overridden_by_config``
    for the dedicated regression test on the URL pairing.
    """
    fake_creds_path = tmp_path / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_creds_path)

    cf = CredentialsFile()
    cf.set_profile("default", _oauth_creds_with_workspace(SAMPLE_CTX_ID))
    save_credentials_file(cf, fake_creds_path)
    reset_state_cache()

    from kagura_memory.cli import _CONTEXT_ID_AUTO

    mock_client = _mock_files_client("upload", _file_object())
    with (
        patch(
            "kagura_memory.cli.load_config",
            return_value={"api_key": "key", "context_id": _CONTEXT_ID_AUTO},
        ),
        patch("kagura_memory.cli.FilesClient") as mock_client_cls,
    ):
        _wire_files_client_mock(mock_client_cls, mock_client)
        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p)])

    reset_state_cache()

    assert result.exit_code == 0, result.output
    mock_client.upload.assert_awaited_once()
    assert mock_client.upload.call_args.kwargs["context_id"] == SAMPLE_CTX_ID
    # Pin the credential source explicitly — an `upload(context_id=…)`
    # assertion alone could pass even if config silently won the api_key
    # race, because the OAuth profile and config both target
    # ``SAMPLE_CTX_ID`` in this fixture. Inspecting the resolved auth
    # locks "OAuth wins entirely" rather than "the right workspace
    # happened to be selected" (per PR #119 review feedback).
    from kagura_memory._auth import _OAuthAuth

    resolved = mock_client_cls._from_resolved_auth.call_args.args[0]
    assert isinstance(resolved, _OAuthAuth), (
        f"OAuth profile should win; got {type(resolved).__name__}"
    )


def test_files_upload_env_wins_over_config_api_key(monkeypatch, tmp_path):
    """``KAGURA_API_KEY`` env + ``.kagura.json`` api_key + ``-c`` → env wins (#118 BREAKING).

    Pre-#118 the Files CLI used a ``config > env > OAuth`` precedence,
    so a session with both env and config api_key set would silently
    pick config. Post-#118 the Files CLI walks the canonical SDK chain
    (``env > OAuth > config``), aligning with ``KaguraClient``. The
    env api_key has no workspace source, so ``-c`` is required —
    asserting it explicitly threads through and the upload proceeds
    locks the new precedence in place against a future revert.
    """
    monkeypatch.setenv("KAGURA_API_KEY", "env-key")

    mock_client = _mock_files_client("upload", _file_object())
    with (
        patch(
            "kagura_memory.cli.load_config",
            return_value={"api_key": "config-key", "context_id": "should-be-ignored"},
        ),
        patch("kagura_memory.cli.FilesClient") as mock_client_cls,
    ):
        _wire_files_client_mock(mock_client_cls, mock_client)
        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p), "-c", SAMPLE_CTX_ID])

    assert result.exit_code == 0, result.output
    mock_client.upload.assert_awaited_once()
    assert mock_client.upload.call_args.kwargs["context_id"] == SAMPLE_CTX_ID
    # FilesClient was constructed via _from_resolved_auth — inspect the
    # _StaticAuth that flowed through to verify the env path won.
    resolved = mock_client_cls._from_resolved_auth.call_args.args[0]
    assert resolved.source == "env"
    assert resolved.api_key == "env-key"


def test_files_upload_rejects_env_api_key_without_context(monkeypatch, tmp_path):
    """``KAGURA_API_KEY`` env + OAuth profile + no ``-c`` → reject (no associated workspace).

    Same-source pairing (issue #115): api_key from ``KAGURA_API_KEY``
    env has no workspace source attached, and falling through to the
    OAuth profile's ``workspace_id`` would mix sources. The CLI must
    require ``--context-id`` explicitly.
    """
    fake_creds_path = tmp_path / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_creds_path)
    monkeypatch.setenv("KAGURA_API_KEY", "env-key")
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)

    cf = CredentialsFile()
    cf.set_profile("default", _oauth_creds_with_workspace(SAMPLE_CTX_ID))
    save_credentials_file(cf, fake_creds_path)
    reset_state_cache()

    with patch("kagura_memory.cli.load_config", return_value={}):
        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p)])

    reset_state_cache()

    assert result.exit_code != 0, result.output
    assert "KAGURA_API_KEY env" in result.output
    assert "--context-id" in result.output


def test_files_upload_oauth_profile_without_workspace_id_errors(monkeypatch, tmp_path):
    """OAuth profile resolved but ``workspace_id`` is empty → actionable error.

    Pre-issue-#170 OAuth profiles did not record ``workspace_id`` (the
    field was added later). An operator upgrading from such a profile
    has a valid login but no bound workspace; the CLI must point them
    at ``kagura auth login`` or ``--context-id``, not produce a 422 at
    request time.
    """
    fake_creds_path = tmp_path / "credentials.json"
    monkeypatch.setattr("kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH", fake_creds_path)
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)

    cf = CredentialsFile()
    # workspace_id="" simulates the legacy profile shape.
    cf.set_profile("default", _oauth_creds_with_workspace(""))
    save_credentials_file(cf, fake_creds_path)
    reset_state_cache()

    with patch("kagura_memory.cli.load_config", return_value={}):
        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p)])

    reset_state_cache()

    assert result.exit_code != 0, result.output
    assert "kagura auth login" in result.output or "--context-id" in result.output


def test_files_upload_env_api_key_with_explicit_context_succeeds(monkeypatch, tmp_path):
    """``KAGURA_API_KEY`` env + ``-c`` → succeed (operator override).

    The mirror of :func:`test_files_upload_rejects_env_api_key_without_context`:
    once the operator supplies ``--context-id`` explicitly, env-derived
    api_key is fine — the explicit flag is the highest-priority
    workspace source regardless of api_key source.
    """
    explicit_ctx = "30000000-0000-0000-0000-000000000004"
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    monkeypatch.setenv("KAGURA_API_KEY", "env-key")
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    reset_state_cache()

    with (
        patch("kagura_memory.cli.load_config", return_value={}),
        patch("kagura_memory.cli.FilesClient") as mock_client_cls,
    ):
        mock_client = _mock_files_client("upload", _file_object())
        _wire_files_client_mock(mock_client_cls, mock_client)

        p = tmp_path / "hello.txt"
        p.write_text("hi")
        runner = CliRunner()
        result = runner.invoke(main, ["files", "upload", str(p), "--context-id", explicit_ctx])

    reset_state_cache()

    assert result.exit_code == 0, result.output
    assert mock_client.upload.call_args.kwargs["context_id"] == explicit_ctx
