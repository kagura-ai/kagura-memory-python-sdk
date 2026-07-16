"""Tests for CLI commands."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from kagura_memory.auth.credentials import reset_state_cache
from kagura_memory.cli import _parse_tags, main
from tests.conftest import sleep_report_summary_dict


@pytest.fixture(autouse=True)
def _isolate_credential_state(tmp_path, monkeypatch):
    """Isolate every test from real ``~/.kagura/credentials.json`` and env vars.

    Several CLI helpers (``_get_resource_client``, ``_run_files_command``)
    now walk the canonical SDK chain (``env > OAuth profile > .kagura.json``)
    via ``_resolve_auth``. Without this isolation, a developer's stored
    OAuth profile or a stale ``KAGURA_PROFILE`` could pre-empt the
    config-only fixtures used by the resource / context tests below
    and make them flaky across machines.
    """
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "default-credentials.json",
    )
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.delenv("KAGURA_MCP_URL", raising=False)
    reset_state_cache()
    yield
    reset_state_cache()


def test_parse_tags():
    """_parse_tags should handle various inputs."""
    assert _parse_tags(None) is None
    assert _parse_tags("") is None
    assert _parse_tags(",,,") is None
    assert _parse_tags("a, b, c") == ["a", "b", "c"]
    assert _parse_tags("single") == ["single"]
    assert _parse_tags(" spaced , tags ") == ["spaced", "tags"]


@patch("kagura_memory.cli.load_config")
def test_config_show(mock_config):
    """config show should display masked API key."""
    mock_config.return_value = {
        "api_key": "kagura_12345678abcdef",
        "mcp_url": "https://memory.kagura-ai.com/mcp",
        "model": "gpt-5.4-nano",
    }
    runner = CliRunner()
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0
    assert "kagura_1...cdef" in result.output
    assert "gpt-5.4-nano" in result.output


@patch("kagura_memory.cli.load_config")
def test_config_show_no_api_key(mock_config):
    """config show should work without API key."""
    mock_config.return_value = {"api_key": "", "mcp_url": "https://test.com/mcp"}
    runner = CliRunner()
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0


@patch("kagura_memory.cli.load_config")
def test_remember_missing_api_key(mock_config, monkeypatch, tmp_path):
    """remember should fail when neither api_key nor OAuth credentials exist."""
    mock_config.return_value = {"api_key": "", "context_id": "ctx-1"}
    # Make sure env vars + credentials.json paths cannot rescue.
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    # Also short-circuit the .kagura.json fallback inside _resolve_auth.
    monkeypatch.setattr("kagura_memory._auth.load_config", lambda: {"api_key": ""})
    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data"])
    assert result.exit_code != 0
    assert "No credentials found" in result.output or "kagura auth login" in result.output


@patch("kagura_memory.cli.load_config")
def test_remember_missing_context(mock_config):
    """remember should fail without context_id."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}
    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data"])
    assert result.exit_code != 0
    assert "context_id required" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_contexts_command(mock_client_cls, mock_config):
    """contexts command should list available contexts."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_contexts.return_value = {"contexts": [{"id": "c1", "name": "test"}]}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["contexts"])
    assert result.exit_code == 0
    assert "c1" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_recall_command(mock_client_cls, mock_config):
    """recall command should search and output results."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }

    mock_client = AsyncMock()
    mock_client.recall.return_value = {"results": [{"summary": "found"}]}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["recall", "test query"])
    assert result.exit_code == 0
    assert "found" in result.output


def test_forget_requires_memory_id_or_query():
    """forget should fail without --memory-id or --query."""
    runner = CliRunner()
    result = runner.invoke(main, ["forget", "-c", "ctx"])
    assert result.exit_code != 0
    assert "Either --memory-id or --query" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_list(mock_client_cls, mock_config):
    """context list should work."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_contexts.return_value = {"contexts": [{"id": "c1"}]}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "list"])
    assert result.exit_code == 0
    assert "c1" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_create(mock_client_cls, mock_config):
    """context create should call create_context."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.create_context.return_value = {"id": "new-uuid", "name": "dev"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "create", "-n", "dev", "-s", "Dev context"])
    assert result.exit_code == 0
    assert "new-uuid" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_update(mock_client_cls, mock_config):
    """context update should call update_context."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_context.return_value = {"id": "uuid-1", "summary": "updated"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1", "-s", "updated"])
    assert result.exit_code == 0
    assert "updated" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_update_lock(mock_client_cls, mock_config):
    """context update --lock should pass is_locked=True."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_context.return_value = {"id": "uuid-1", "is_locked": True}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1", "--lock"])
    assert result.exit_code == 0
    call_kwargs = mock_client.update_context.call_args[1]
    assert call_kwargs["is_locked"] is True


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_update_unlock(mock_client_cls, mock_config):
    """context update --unlock should pass is_locked=False."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_context.return_value = {"id": "uuid-1", "is_locked": False}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1", "--unlock"])
    assert result.exit_code == 0
    call_kwargs = mock_client.update_context.call_args[1]
    assert call_kwargs["is_locked"] is False


def test_context_update_requires_option():
    """context update should fail without any update option."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1"])
    assert result.exit_code != 0
    assert "At least one update option" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_update_memory_by_id(mock_client_cls, mock_config):
    """update-memory with --memory-id should call update_memory."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }

    mock_client = AsyncMock()
    mock_client.update_memory.return_value = {"status": "success"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-m", "mem-1", "-s", "updated summary"])
    assert result.exit_code == 0
    mock_client.update_memory.assert_called_once()


def test_update_memory_requires_id():
    """update-memory without --memory-id or --external-id should fail."""
    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-s", "test"])
    assert result.exit_code != 0
    assert "Either --memory-id or --external-id" in result.output


def test_update_memory_rejects_both_ids():
    """update-memory with both --memory-id and --external-id should fail."""
    runner = CliRunner()
    result = runner.invoke(main, ["update-memory", "-m", "mem-1", "--external-id", "ext-1"])
    assert result.exit_code != 0
    assert "only one" in result.output.lower()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_delete(mock_client_cls, mock_config):
    """context delete should call delete_context with -y flag."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.delete_context.return_value = {
        "status": "success",
        "message": "Context 'test' has been soft-deleted.",
        "context_id": "uuid-1",
        "context_name": "test",
    }
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["context", "delete", "uuid-1", "-y"])
    assert result.exit_code == 0
    mock_client.delete_context.assert_called_once_with(context_id="uuid-1")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_delete_prompts_confirmation(mock_client_cls, mock_config):
    """context delete without -y should prompt for confirmation."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "delete", "uuid-1"], input="n\n")
    assert result.exit_code != 0


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_context_search_config(mock_client_cls, mock_config):
    """context search-config should call update_search_config."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_search_config.return_value = {"status": "success"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main, ["context", "search-config", "uuid-1", "--semantic", "0.5", "--bm25", "0.5"]
    )
    assert result.exit_code == 0
    assert "success" in result.output


def test_context_search_config_requires_option():
    """context search-config should fail without options."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "search-config", "uuid-1"])
    assert result.exit_code != 0
    assert "At least one option" in result.output


def test_context_search_config_invalid_weight_sum():
    """context search-config should reject weights that don't sum to 1.0."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["context", "search-config", "uuid-1", "--semantic", "0.5", "--bm25", "0.3"]
    )
    assert result.exit_code != 0
    assert "must sum to 1.0" in result.output


def test_context_search_config_invalid_range():
    """context search-config should reject out-of-range values."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "search-config", "uuid-1", "--semantic", "1.5"])
    assert result.exit_code != 0


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_with_empty_tags(mock_client_cls, mock_config):
    """remember with empty tags should not pass empty strings."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }

    mock_client = AsyncMock()
    mock_client.remember.return_value = {"memory_id": "m1"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data", "--tags", ",,,"])
    assert result.exit_code == 0
    # tags should be None (filtered out), not ['', '', '']
    call_kwargs = mock_client.remember.call_args
    assert call_kwargs[1].get("tags") is None or call_kwargs[1].get("tags") == []


def _remember_mock_client(mock_client_cls):
    """Wire a mock KaguraClient whose ``remember`` returns a memory id."""
    mock_client = AsyncMock()
    mock_client.remember.return_value = {"memory_id": "m1"}
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client
    return mock_client


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_passes_source_uri_and_type(mock_client_cls, mock_config):
    """remember should forward --source-uri / --source-type to the client."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "remember",
            "-s",
            "test",
            "--content",
            "data",
            "--source-uri",
            "file:///foo.md",
            "--source-type",
            "file",
        ],
    )
    assert result.exit_code == 0
    kwargs = mock_client.remember.call_args[1]
    assert kwargs["source_uri"] == "file:///foo.md"
    assert kwargs["source_type"] == "file"


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_parses_linked_memory_ids_with_whitespace(mock_client_cls, mock_config):
    """--linked-memory-ids should split on comma and strip whitespace."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "remember",
            "-s",
            "test",
            "--content",
            "data",
            "--linked-memory-ids",
            "uuid-1, uuid-2 , uuid-3",
        ],
    )
    assert result.exit_code == 0
    kwargs = mock_client.remember.call_args[1]
    assert kwargs["linked_memory_ids"] == ["uuid-1", "uuid-2", "uuid-3"]


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_parses_linked_source_uris(mock_client_cls, mock_config):
    """--linked-source-uris should split on comma into a list."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "remember",
            "-s",
            "test",
            "--content",
            "data",
            "--linked-source-uris",
            "file:///a.md,vault://v/b",
        ],
    )
    assert result.exit_code == 0
    kwargs = mock_client.remember.call_args[1]
    assert kwargs["linked_source_uris"] == ["file:///a.md", "vault://v/b"]


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_remember_omits_provenance_when_not_given(mock_client_cls, mock_config):
    """Without provenance flags, remember must not stamp source_type/uri/links.

    Guards against a silent behavior change: a plain ``kagura remember`` should
    pass None for these so the server stores no misleading provenance.
    """
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "context_id": "ctx",
    }
    mock_client = _remember_mock_client(mock_client_cls)

    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data"])
    assert result.exit_code == 0
    kwargs = mock_client.remember.call_args[1]
    assert kwargs["source_uri"] is None
    assert kwargs["source_type"] is None
    assert kwargs["linked_memory_ids"] is None
    assert kwargs["linked_source_uris"] is None


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_stats(mock_rc_cls, mock_config):
    """resource stats should call get_resource_impact."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    json_out = '{"token_count": 2, "memory_count": 50}'
    mock_rc.get_resource_impact.return_value = MagicMock(
        model_dump_json=lambda indent=None: json_out
    )
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "stats", "-r", "products"])
    assert result.exit_code == 0
    assert "token_count" in result.output
    mock_rc.get_resource_impact.assert_called_once_with("products")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_schema(mock_rc_cls, mock_config):
    """resource schema should call get_resource_schema."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    json_out = '{"resource_id": "products", "schema_version": 1}'
    mock_rc.get_resource_schema.return_value = MagicMock(
        model_dump_json=lambda indent=None: json_out
    )
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "schema", "-r", "products"])
    assert result.exit_code == 0
    assert "schema_version" in result.output
    mock_rc.get_resource_schema.assert_called_once_with("products", schema_version=None)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_schema_not_registered(mock_rc_cls, mock_config):
    """resource schema should show message when schema is not registered."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.get_resource_schema.return_value = None
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "schema", "-r", "no-schema"])
    assert result.exit_code == 0
    assert "No schema registered" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_setup(mock_rc_cls, mock_config):
    """resource setup should call setup_resource."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.setup_resource.return_value = MagicMock(
        model_dump_json=lambda indent=None: '{"token": "kagura_resource_abc", "id": 1}'
    )
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "setup", "-r", "products", "-s", "catalog"])
    assert result.exit_code == 0
    assert "kagura_resource_abc" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_list(mock_rc_cls, mock_config):
    """resource list should call list_resources."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    json_out = '{"resources": [], "total": 0}'
    mock_rc.list_resources.return_value = MagicMock(model_dump_json=lambda indent=None: json_out)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "list"])
    assert result.exit_code == 0
    assert '"total": 0' in result.output
    mock_rc.list_resources.assert_called_once_with()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_indexer_status(mock_rc_cls, mock_config):
    """resource indexer-status should call get_indexer_status."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    json_out = '{"resource_id": "products", "state": null, "recent_events": []}'
    mock_rc.get_indexer_status.return_value = MagicMock(
        model_dump_json=lambda indent=None: json_out
    )
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "indexer-status", "-r", "products"])
    assert result.exit_code == 0
    assert '"resource_id"' in result.output
    mock_rc.get_indexer_status.assert_called_once_with("products")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_csv(mock_rc_cls, mock_config):
    """resource import should parse CSV and batch ingest."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.ingest_events.return_value = MagicMock(created_count=3, failed_count=0)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    csv_content = "name,price\nWidget,9.99\nGadget,19.99\nThing,29.99"
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "products", "-k", "TOKEN", "--format", "csv"],
        input=csv_content,
    )
    assert result.exit_code == 0
    assert '"created": 3' in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_jsonl(mock_rc_cls, mock_config):
    """resource import should parse JSONL."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.ingest_events.return_value = MagicMock(created_count=2, failed_count=0)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    jsonl = '{"name":"A"}\n{"name":"B"}'
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "jsonl"],
        input=jsonl,
    )
    assert result.exit_code == 0
    assert '"created": 2' in result.output


def test_resource_import_no_format_stdin():
    """resource import from stdin without --format should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY"],
        input="some data",
    )
    assert result.exit_code != 0
    assert "Cannot detect format" in result.output


def test_resource_import_bad_id_column():
    """resource import with nonexistent --id-column should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "csv", "--id-column", "nope"],
        input="name,price\nWidget,9.99",
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_resource_import_json_non_object_items():
    """resource import should reject JSON array with non-object items."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input="[1, 2, 3]",
    )
    assert result.exit_code != 0
    assert "not an object" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.ResourceClient")
def test_resource_import_json(mock_rc_cls, mock_config):
    """resource import should parse JSON array."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_rc = AsyncMock()
    mock_rc.ingest_events.return_value = MagicMock(created_count=2, failed_count=0)
    mock_rc.__aenter__ = AsyncMock(return_value=mock_rc)
    mock_rc.__aexit__ = AsyncMock(return_value=None)
    mock_rc_cls.from_mcp_url.return_value = mock_rc
    mock_rc_cls._from_resolved_auth.return_value = mock_rc

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input='[{"name":"A"},{"name":"B"}]',
    )
    assert result.exit_code == 0
    assert '"created": 2' in result.output


def test_resource_ingest_invalid_payload_surfaces_json_error():
    """`resource ingest -p '<not json>'` translates JSONDecodeError via _exc_message (#130)."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "resource",
            "ingest",
            "-r",
            "products",
            "-k",
            "TOKEN",
            "--doc-id",
            "SKU-1",
            "-p",
            "not json",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid JSON payload" in result.output


def test_resource_ingest_batch_invalid_json_file_surfaces_error(tmp_path):
    """`resource ingest-batch -f <bad json>` translates JSONDecodeError via _exc_message (#130)."""
    bad = tmp_path / "events.json"
    bad.write_text("not json")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "ingest-batch", "-r", "res", "-k", "TOKEN", "-f", str(bad)],
    )
    assert result.exit_code != 0
    assert "Invalid JSON file" in result.output


def test_resource_import_read_failure_surfaces_class_name(monkeypatch):
    """`resource import` translates a read() OSError into ClickException via _exc_message (#130)."""
    import io

    class _RaisingFile(io.StringIO):
        def read(self, *_args, **_kwargs):
            raise OSError("disk gone")

    runner = CliRunner()
    # Inject a file-like that raises on read by monkeypatching click.File.convert
    # to swap the wrapped file just before resource_import reads it.
    real_convert = __import__("click").File.convert

    def _fake_convert(self, value, param, ctx):
        if isinstance(value, str):
            return _RaisingFile("placeholder")
        return real_convert(self, value, param, ctx)

    monkeypatch.setattr("click.File.convert", _fake_convert)

    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "TOKEN", "--format", "json", "-f", "any.json"],
    )
    assert result.exit_code != 0
    assert "Failed to read input" in result.output
    assert "disk gone" in result.output


def test_resource_import_invalid_json():
    """resource import with invalid JSON should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input="not json",
    )
    assert result.exit_code != 0
    assert "Invalid JSON" in result.output


def test_resource_import_json_not_array():
    """resource import with JSON object (not array) should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input='{"not": "array"}',
    )
    assert result.exit_code != 0
    assert "must be an array" in result.output


def test_resource_import_invalid_jsonl():
    """resource import with invalid JSONL line should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "jsonl"],
        input='{"ok":1}\nnot json\n{"ok":2}',
    )
    assert result.exit_code != 0
    assert "Invalid JSONL at line 2" in result.output


def test_resource_import_empty_data():
    """resource import with empty input should error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "jsonl"],
        input="",
    )
    assert result.exit_code != 0
    assert "No data found" in result.output


def test_resource_import_auto_detect_csv(tmp_path):
    """resource import should auto-detect CSV from extension."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name\nWidget")

    runner = CliRunner()
    # Will fail at ResourceClient but format detection should pass
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "-f", str(csv_file)],
    )
    # It gets past format detection (would fail at connection)
    assert "Cannot detect format" not in (result.output or "")


# ============================================================================
# Sleep Maintenance CLI (issue #85)
# ============================================================================


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_history(mock_client_cls, mock_config):
    """`kagura sleep history` echoes a JSON object with a top-level ``reports`` array."""
    from kagura_memory import SleepReport

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.get_sleep_history.return_value = [SleepReport(**sleep_report_summary_dict("rid-9"))]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "history", "ctx-1", "--limit", "5"])
    assert result.exit_code == 0, result.output
    assert "rid-9" in result.output
    mock_client.get_sleep_history.assert_called_once_with(context_id="ctx-1", limit=5)


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_report(mock_client_cls, mock_config):
    """`kagura sleep report` echoes the flattened detail JSON."""
    from kagura_memory import SleepReportDetail

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    detail = SleepReportDetail(
        **sleep_report_summary_dict("rid-9"),
        memories_flagged=0,
        embedding_calls_made=2,
        actions=[],
        action_count=0,
    )
    mock_client = AsyncMock()
    mock_client.get_sleep_report.return_value = detail
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "report", "ctx-1", "rid-9"])
    assert result.exit_code == 0, result.output
    assert "rid-9" in result.output
    assert "memories_flagged" in result.output
    mock_client.get_sleep_report.assert_called_once_with(context_id="ctx-1", report_id="rid-9")


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_rollback_with_yes_flag(mock_client_cls, mock_config):
    """`kagura sleep rollback -y` skips both the prompt and the pre-fetch."""
    from kagura_memory import RollbackResult, RollbackSummary

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    rollback = RollbackResult(
        report_id="rid-9",
        status="rolled_back",
        rollback_summary=RollbackSummary(edges_deleted=2, merges_reversed=1),
    )

    mock_client = AsyncMock()
    mock_client.rollback_sleep_run.return_value = rollback
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code == 0, result.output
    assert "rolled_back" in result.output
    mock_client.rollback_sleep_run.assert_called_once_with(context_id="ctx-1", report_id="rid-9")
    # --yes must skip the cosmetic pre-fetch — no wasted round trip.
    mock_client.get_sleep_report.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_rollback_aborts_on_no(mock_client_cls, mock_config):
    """Without -y, answering 'n' aborts before calling rollback_sleep_run."""
    from kagura_memory import SleepReportDetail

    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    detail = SleepReportDetail(
        **sleep_report_summary_dict("rid-9"),
        memories_flagged=0,
        embedding_calls_made=2,
        actions=[],
        action_count=3,
    )
    mock_client = AsyncMock()
    mock_client.get_sleep_report.return_value = detail
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "rollback", "ctx-1", "rid-9"], input="n\n")
    assert result.exit_code != 0
    mock_client.rollback_sleep_run.assert_not_called()


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_sleep_rollback_wraps_unexpected_exception(mock_client_cls, mock_config):
    """Unexpected exceptions inside _run() surface as click.ClickException."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.rollback_sleep_run.side_effect = RuntimeError("boom")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code != 0
    assert "Error: boom" in result.output


@patch("kagura_memory.cli.load_config")
def test_get_kagura_client_missing_api_key(mock_config, monkeypatch, tmp_path):
    """_get_kagura_client surfaces a credential error when no api_key + no OAuth."""
    mock_config.return_value = {"api_key": "", "mcp_url": "https://test.com/mcp"}
    monkeypatch.delenv("KAGURA_API_KEY", raising=False)
    monkeypatch.delenv("KAGURA_PROFILE", raising=False)
    monkeypatch.setattr(
        "kagura_memory.auth.credentials.DEFAULT_CREDENTIALS_PATH",
        tmp_path / "missing-credentials.json",
    )
    monkeypatch.setattr("kagura_memory._auth.load_config", lambda: {"api_key": ""})

    runner = CliRunner()
    result = runner.invoke(main, ["sleep", "rollback", "ctx-1", "rid-9", "-y"])
    assert result.exit_code != 0
    assert "No credentials found" in result.output or "kagura auth login" in result.output


# ============================================================================
# Edge CRUD CLI tests
# ============================================================================


def _edge_obj():
    """Build an Edge model instance for CLI test mocking."""
    from datetime import datetime

    from kagura_memory import Edge

    return Edge(
        source_id="src-uuid",
        target_id="tgt-uuid",
        edge_type="related_to",
        weight=0.5,
        confidence=1.0,
        created_at=datetime(2026, 4, 29, 0, 0, 0),
        last_updated=datetime(2026, 4, 29, 0, 5, 0),
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_list(mock_client_cls, mock_config):
    """edge list should call list_edges and emit JSON."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_edges.return_value = [_edge_obj()]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["edge", "list", "ctx-1", "mem-1"])
    assert result.exit_code == 0, result.output
    assert "src-uuid" in result.output
    assert "tgt-uuid" in result.output

    mock_client.list_edges.assert_called_once_with(
        context_id="ctx-1",
        memory_id="mem-1",
        min_weight=0.0,
        edge_types=None,
        limit=None,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_list_with_filters(mock_client_cls, mock_config):
    """edge list should pass --min-weight, --type, --limit through."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.list_edges.return_value = []
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "edge",
            "list",
            "ctx-1",
            "mem-1",
            "--min-weight",
            "0.5",
            "--type",
            "related_to,depends_on",
            "--limit",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.list_edges.assert_called_once_with(
        context_id="ctx-1",
        memory_id="mem-1",
        min_weight=0.5,
        edge_types=["related_to", "depends_on"],
        limit=10,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_create(mock_client_cls, mock_config):
    """edge create should call create_edge with defaults."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.create_edge.return_value = _edge_obj()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["edge", "create", "ctx-1", "src-uuid", "tgt-uuid"])
    assert result.exit_code == 0, result.output
    assert "related_to" in result.output

    mock_client.create_edge.assert_called_once_with(
        context_id="ctx-1",
        source_id="src-uuid",
        target_id="tgt-uuid",
        edge_type="related_to",
        weight=0.5,
        confidence=1.0,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_create_with_options(mock_client_cls, mock_config):
    """edge create should pass --type, --weight, --confidence."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.create_edge.return_value = _edge_obj()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "edge",
            "create",
            "ctx-1",
            "src-uuid",
            "tgt-uuid",
            "--type",
            "depends_on",
            "--weight",
            "0.8",
            "--confidence",
            "0.9",
        ],
    )
    assert result.exit_code == 0, result.output
    mock_client.create_edge.assert_called_once_with(
        context_id="ctx-1",
        source_id="src-uuid",
        target_id="tgt-uuid",
        edge_type="depends_on",
        weight=0.8,
        confidence=0.9,
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_update(mock_client_cls, mock_config):
    """edge update should call update_edge with the provided fields."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.update_edge.return_value = _edge_obj()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["edge", "update", "ctx-1", "src-uuid", "tgt-uuid", "--weight", "0.9"],
    )
    assert result.exit_code == 0, result.output

    mock_client.update_edge.assert_called_once_with(
        context_id="ctx-1",
        source_id="src-uuid",
        target_id="tgt-uuid",
        weight=0.9,
        edge_type=None,
    )


def test_edge_update_requires_option():
    """edge update should fail without --weight or --type."""
    runner = CliRunner()
    result = runner.invoke(main, ["edge", "update", "ctx-1", "src-uuid", "tgt-uuid"])
    assert result.exit_code != 0
    assert "At least one of --weight or --type" in result.output


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_delete_with_yes_flag(mock_client_cls, mock_config):
    """edge delete with -y should call delete_edge without prompting."""
    mock_config.return_value = {"api_key": "key", "mcp_url": "https://test.com/mcp"}

    mock_client = AsyncMock()
    mock_client.delete_edge.return_value = True
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client_cls.return_value = mock_client

    runner = CliRunner()
    result = runner.invoke(main, ["edge", "delete", "ctx-1", "src-uuid", "tgt-uuid", "-y"])
    assert result.exit_code == 0, result.output
    assert "true" in result.output.lower()

    mock_client.delete_edge.assert_called_once_with(
        context_id="ctx-1",
        source_id="src-uuid",
        target_id="tgt-uuid",
    )


@patch("kagura_memory.cli.load_config")
@patch("kagura_memory.cli.KaguraClient")
def test_edge_delete_prompts_confirmation(mock_client_cls, mock_config):
    """edge delete without -y should prompt and abort on 'n'."""
    runner = CliRunner()
    result = runner.invoke(main, ["edge", "delete", "ctx-1", "src-uuid", "tgt-uuid"], input="n\n")
    assert result.exit_code != 0
