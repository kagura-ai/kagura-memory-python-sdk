"""Tests for CLI commands."""

from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from kagura_memory.cli import main


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
def test_remember_missing_api_key(mock_config):
    """remember should fail without API key."""
    mock_config.return_value = {"api_key": ""}
    runner = CliRunner()
    result = runner.invoke(main, ["remember", "-s", "test", "--content", "data"])
    assert result.exit_code != 0
    assert "No API key" in result.output


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


def test_context_update_requires_option():
    """context update should fail without any update option."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "update", "uuid-1"])
    assert result.exit_code != 0
    assert "At least one update option" in result.output


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
@patch("kagura_memory.cli.KaguraAgent")
def test_process_command(mock_agent_cls, mock_config):
    """process command should use async with and return result."""
    mock_config.return_value = {
        "api_key": "key",
        "mcp_url": "https://test.com/mcp",
        "model": "gpt-test",
        "context_id": "ctx",
    }

    mock_agent = AsyncMock()
    mock_agent.process.return_value = MagicMock(
        model_dump=lambda: {"remembered": [], "recalled": [], "context_used": "ctx"}
    )
    mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
    mock_agent.__aexit__ = AsyncMock(return_value=None)
    mock_agent_cls.return_value = mock_agent

    runner = CliRunner()
    result = runner.invoke(main, ["process", "-m", "test message"])
    assert result.exit_code == 0
    assert "ctx" in result.output


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

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "stats", "-r", "products"])
    assert result.exit_code == 0
    assert "token_count" in result.output
    mock_rc.get_resource_impact.assert_called_once_with("products")


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

    runner = CliRunner()
    result = runner.invoke(main, ["resource", "setup", "-r", "products", "-s", "catalog"])
    assert result.exit_code == 0
    assert "kagura_resource_abc" in result.output


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

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["resource", "import", "-r", "res", "-k", "KEY", "--format", "json"],
        input='[{"name":"A"},{"name":"B"}]',
    )
    assert result.exit_code == 0
    assert '"created": 2' in result.output


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
