"""Configuration loading for Kagura Memory SDK."""

import json
import os
from pathlib import Path
from typing import Any


def load_config() -> dict[str, Any]:
    """
    Load configuration from .kagura.json or environment variables.

    Search order:
    1. ./.kagura.json (current directory)
    2. ~/.kagura.json (home directory)
    3. Environment variables (KAGURA_API_KEY, KAGURA_MCP_URL, KAGURA_MODEL)

    Returns:
        Configuration dictionary with keys: api_key, mcp_url, model, context_id, llm_api_key
    """
    config = {}

    # 1. Try current directory
    local_config = Path(".kagura.json")
    if local_config.exists():
        try:
            config = json.loads(local_config.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Invalid JSON or encoding in .kagura.json (expected UTF-8): {e}"
            ) from e
    # 2. Try home directory if not found locally
    elif (home_config := Path.home() / ".kagura.json").exists():
        try:
            config = json.loads(home_config.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Invalid JSON or encoding in ~/.kagura.json (expected UTF-8): {e}"
            ) from e
    # 3. Fall back to environment variables
    else:
        config = {
            "api_key": os.getenv("KAGURA_API_KEY", ""),
            "mcp_url": os.getenv("KAGURA_MCP_URL", "https://memory.kagura-ai.com/mcp"),
            "model": os.getenv("KAGURA_MODEL", "gpt-5.4-nano"),
            "context_id": os.getenv("KAGURA_CONTEXT_ID"),
        }

    # C-2: Don't write LLM API key to os.environ (leaks to subprocesses).
    # Instead, keep llm_api_key in config dict and pass it explicitly
    # to litellm.acompletion(api_key=...) in agent.py.

    return config
