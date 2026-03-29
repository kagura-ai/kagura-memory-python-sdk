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
