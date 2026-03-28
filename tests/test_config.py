"""Tests for configuration loading."""

import json
from unittest.mock import patch

import pytest

from kagura_memory.config import load_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove Kagura env vars to isolate tests."""
    for key in ("KAGURA_API_KEY", "KAGURA_MCP_URL", "KAGURA_MODEL", "KAGURA_CONTEXT_ID"):
        monkeypatch.delenv(key, raising=False)


def _mock_path(exists: bool, content: str = "{}"):
    """Create a mock Path that exists/doesn't and returns content."""
    from unittest.mock import MagicMock

    p = MagicMock()
    p.exists.return_value = exists
    p.read_text.return_value = content
    return p


def test_load_from_local_file():
    """Should load config from ./.kagura.json."""
    config_data = {"api_key": "local_key", "model": "gpt-4"}
    local = _mock_path(True, json.dumps(config_data))
    home = _mock_path(False)

    with patch("kagura_memory.config.Path") as mock_path_cls:
        mock_path_cls.return_value = local
        mock_path_cls.home.return_value.__truediv__ = lambda self, x: home

        result = load_config()

    assert result["api_key"] == "local_key"
    assert result["model"] == "gpt-4"


def test_load_from_home_file():
    """Should load from ~/.kagura.json if local not found."""
    config_data = {"api_key": "home_key"}
    local = _mock_path(False)
    home = _mock_path(True, json.dumps(config_data))

    with patch("kagura_memory.config.Path") as mock_path_cls:
        mock_path_cls.return_value = local
        mock_path_cls.home.return_value.__truediv__ = lambda self, x: home

        result = load_config()

    assert result["api_key"] == "home_key"


def test_load_from_env_vars(monkeypatch):
    """Should fall back to environment variables."""
    monkeypatch.setenv("KAGURA_API_KEY", "env_key")
    monkeypatch.setenv("KAGURA_MCP_URL", "https://custom.com/mcp")
    monkeypatch.setenv("KAGURA_MODEL", "claude-3")
    monkeypatch.setenv("KAGURA_CONTEXT_ID", "my-ctx")

    local = _mock_path(False)
    home = _mock_path(False)

    with patch("kagura_memory.config.Path") as mock_path_cls:
        mock_path_cls.return_value = local
        mock_path_cls.home.return_value.__truediv__ = lambda self, x: home

        result = load_config()

    assert result["api_key"] == "env_key"
    assert result["mcp_url"] == "https://custom.com/mcp"
    assert result["model"] == "claude-3"
    assert result["context_id"] == "my-ctx"


def test_default_values():
    """Should use defaults when no env vars set."""
    local = _mock_path(False)
    home = _mock_path(False)

    with patch("kagura_memory.config.Path") as mock_path_cls:
        mock_path_cls.return_value = local
        mock_path_cls.home.return_value.__truediv__ = lambda self, x: home

        result = load_config()

    assert result["api_key"] == ""
    assert result["mcp_url"] == "https://memory.kagura-ai.com/mcp"
    assert result["model"] == "gpt-5.4-nano"
    assert result["context_id"] is None


def test_invalid_json_local():
    """Should raise ValueError on invalid JSON in local config."""
    local = _mock_path(True, "{invalid json")

    with patch("kagura_memory.config.Path") as mock_path_cls:
        mock_path_cls.return_value = local

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config()


def test_invalid_json_home():
    """Should raise ValueError on invalid JSON in home config."""
    local = _mock_path(False)
    home = _mock_path(True, "not json")

    with patch("kagura_memory.config.Path") as mock_path_cls:
        mock_path_cls.return_value = local
        mock_path_cls.home.return_value.__truediv__ = lambda self, x: home

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config()
