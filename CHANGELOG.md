# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

- `KaguraAgent` now supports Ollama Cloud: new `ollama_api_key` kwarg defaulting to `OLLAMA_API_KEY` env var; `ollama_base_url` defaults to `OLLAMA_HOST` env (then `http://localhost:11434`); `_call_ollama` injects `Authorization: Bearer <key>` when a key is configured. HTTP 401/403 from Ollama is now mapped to `KaguraAuthError` (was generic `KaguraLLMError`). Explicit `ollama_base_url=""` raises `ValueError` instead of producing a relative URL. (#124)
- Ingest `OllamaProvider` now uses the `ollama_chat/` litellm prefix so system messages are preserved through the native `/api/chat` route. Explicit `text_model` / `vision_model` values starting with the legacy `ollama/` prefix are auto-rewritten with a `DeprecationWarning`. Existing local-Ollama setups continue to work unchanged; for Ollama Cloud, set `OLLAMA_API_KEY` and `OLLAMA_API_BASE=https://ollama.com` (read by litellm directly). (#124)
- `kagura auth login` now requests `memory:read memory:write` by default; pass `--read-only` for read-only scope. `--scope` still accepts a custom set. (#122)
- `kagura auth login` falls back to platform openers (`wslview` or `rundll32.exe url.dll,FileProtocolHandler` on WSL, `open` on macOS, `xdg-open` on Linux) when `webbrowser.open()` doesn't actually launch a browser. On WSL the stdlib path is skipped entirely because it can report success without launching a browser. (#122, #125)
- Removed the pre-login web-UI tip from the device-flow prompt — `memory-cloud#772` shipped a login-gated `/device` page that no longer needs the client-side workaround. (#128)
- `kagura setup claude` now falls back to the exception class name when an inner step raises without a message, so the CLI never prints a bare `Setup failed:` (or `Connection failed:`) with nothing after the colon. (#127)
- Extracted the defensive `str(e) or e.__class__.__name__` pattern into a shared `_exc_message` helper and applied it to every `ClickException` and `KaguraConnectionError` wrapper across the SDK (`auth/cli.py`, `auth/device_flow.py`, `cli.py`, `client.py`, `files_client.py`, `resource_client.py`, `setup_claude.py`). Unmessaged exceptions now consistently surface the class name instead of stranding a bare `<prefix>:` in user-facing output. (#130)
