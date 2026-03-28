"""Tests for CLI commands."""

from unittest.mock import AsyncMock, patch

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
