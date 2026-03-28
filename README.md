# Kagura Memory SDK

[![PyPI version](https://img.shields.io/pypi/v/kagura-memory)](https://pypi.org/project/kagura-memory/)
[![Python versions](https://img.shields.io/pypi/pyversions/kagura-memory)](https://pypi.org/project/kagura-memory/)
[![CI](https://github.com/kagura-ai/kagura-memory-python-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/kagura-ai/kagura-memory-python-sdk/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kagura-ai/kagura-memory-python-sdk/graph/badge.svg)](https://codecov.io/gh/kagura-ai/kagura-memory-python-sdk)
[![License: MIT](https://img.shields.io/pypi/l/kagura-memory)](https://github.com/kagura-ai/kagura-memory-python-sdk/blob/main/LICENSE)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

AI-driven memory management for Kagura Memory Cloud.

## Installation

```bash
pip install kagura-memory
# or
uv add kagura-memory
```

For development:

```bash
git clone https://github.com/kagura-ai/kagura-memory-python-sdk.git
cd kagura-memory-python-sdk
uv sync --dev
```

## Quick Start

### Python SDK

```python
from kagura_memory import KaguraAgent, Session, Message

# Initialize agent
agent = KaguraAgent(
    api_key="your_kagura_api_key",
    model="gpt-5.4-nano",
)

# Create a session
session = Session(
    messages=[
        Message(role="user", content="FastAPIでOAuth2を実装したい"),
        Message(role="assistant", content="Authlibを使うパターンが推奨です..."),
        Message(role="user", content="なるほど、これ覚えておいて"),
    ]
)

# Process the session (AI automatically decides what to remember/recall)
result = await agent.process(session, verbose=2)

print(f"Remembered: {len(result.remembered)}")
print(f"Recalled: {len(result.recalled)}")
print(f"Explored: {len(result.explored)}")
print(f"Context: {result.context_used}")
if result.llm_usage:
    print(f"Tokens: {result.llm_usage.total_tokens}")
```

### CLI

#### Configuration

Create `.kagura.json`:

```json
{
  "mcp_url": "https://memory.kagura-ai.com/mcp",
  "api_key": "your_kagura_api_key",
  "model": "gpt-5.4-nano",
  "context_id": "dev",
  "llm_api_key": "your_openai_or_anthropic_api_key"
}
```

**Note on LLM API Keys**:
- `llm_api_key` in `.kagura.json` is optional
- If not provided, LiteLLM will use standard environment variables:
  - OpenAI: `OPENAI_API_KEY`
  - Claude: `ANTHROPIC_API_KEY`
  - Gemini: `GEMINI_API_KEY`

Or use environment variables:

```bash
export KAGURA_API_KEY="your_kagura_api_key"
export KAGURA_MCP_URL="https://memory.kagura-ai.com/mcp"
export KAGURA_MODEL="gpt-5.4-nano"
export OPENAI_API_KEY="your_openai_key"  # For LLM
```

#### Usage

```bash
# AI-powered processing (auto-decides what to remember/recall)
kagura process -m "Remember: FastAPI uses Depends() for DI"
kagura process -m "FastAPIの実装パターンを探して" --deep
kagura process -m "OAuth2について教えて" -vv  # verbose

# Direct memory operations (no LLM required)
kagura remember -s "FastAPI DI pattern" --content "Use Depends()..."
kagura remember -c dev -s "OAuth2 setup" --content "..." --tags "auth,oauth"

kagura recall "FastAPI dependency injection"
kagura recall "OAuth2 implementation" -k 10

kagura explore -m "memory-uuid-here" --depth 3
kagura reference -m "memory-uuid-here"

# Delete memories (soft delete, 30-day recovery)
kagura forget -m "memory-uuid-here"
kagura forget -q "outdated test data" -k 5

# List available contexts
kagura contexts

# Show current config
kagura config show
```

### Claude Code Integration

You can use Kagura Memory as an MCP server in Claude Code. Copy `.mcp.json.example` to `.mcp.json` and fill in your credentials:

```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json with your workspace ID and API key
```

Or use the CLI via Bash:

```bash
# In Claude Code, use Bash tool:
kagura process -m "今日の学び：FastAPIのDIはDepends()を使う"
```

## Features

### Current Version (0.2.2)

- ✅ **LLM-Powered Analysis**: Automatically decides what to remember/recall
- ✅ **Session-Based Input**: Messages + artifacts (code, documents, errors)
- ✅ **Deep Mode** (`deep=True`): Neural Memory graph exploration
- ✅ **Verbose Logging** (0-3): Silent to debug with Rich panels
- ✅ **Context Auto-Selection** (`context_id="auto"`): LLM selects best context
- ✅ **Multiple LLM Support**: OpenAI, Claude, Gemini, Ollama via LiteLLM
- ✅ **Type Safety**: Full Pydantic validation
- ✅ **CLI Commands**: Full suite of commands for AI and direct operations
- ✅ **Graceful Degradation**: Continues even if LLM fails

### New in v0.2.2 (Phase 3 - CLI)

- ✅ **Direct CLI Commands**: `kagura remember`, `kagura recall`, `kagura forget`, `kagura explore`, `kagura reference`, `kagura contexts`
- ✅ **No LLM Required**: Direct memory operations without AI analysis
- ✅ **Flexible Context**: Use `--context-id` or configure in `.kagura.json`

### v0.2.1 (Phase 2.5)

- ✅ **Dynamic Tool Definitions**: Fetches MCP tool specifications via `tools/list`
- ✅ **Enhanced Prompts**: LLM receives actual parameter schemas and context info
- ✅ **Intelligent Caching**: 5-minute TTL cache for tool/context definitions
- ✅ **Automatic Fallback**: Uses static prompts if dynamic fetching fails

## Supported LLM Models

Via [LiteLLM](https://github.com/BerriAI/litellm):

```python
# OpenAI
agent = KaguraAgent(api_key="...", model="gpt-5.4-nano")

# Claude
agent = KaguraAgent(api_key="...", model="claude-sonnet-4-20250514")

# Gemini
agent = KaguraAgent(api_key="...", model="gemini/gemini-1.5-flash")

# Ollama (local)
agent = KaguraAgent(api_key="...", model="ollama/llama3")
```

## Development

### Setup

```bash
uv sync --dev
```

### Quality Checks

```bash
uv run ruff check src/ tests/   # Lint
uv run ruff format src/ tests/  # Format
uv run pyright src/              # Type check
uv run pytest tests/ -v          # Test
```

## Links

- [Kagura Memory Cloud](https://github.com/kagura-ai/memory-cloud)
- [SDK Issue Tracker](https://github.com/kagura-ai/kagura-memory-python-sdk/issues)

## License

MIT License - see LICENSE file for details.
