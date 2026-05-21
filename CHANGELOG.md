# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

- `kagura auth login` now requests `memory:read memory:write` by default; pass `--read-only` for read-only scope. `--scope` still accepts a custom set. (#122)
- `kagura auth login` falls back to platform openers (`wslview` or `rundll32.exe url.dll,FileProtocolHandler` on WSL, `open` on macOS, `xdg-open` on Linux) when `webbrowser.open()` doesn't actually launch a browser. On WSL the stdlib path is skipped entirely because it can report success without launching a browser. (#122, #125)
- Removed the pre-login web-UI tip from the device-flow prompt — `memory-cloud#772` shipped a login-gated `/device` page that no longer needs the client-side workaround. (#128)
- `kagura setup claude` now falls back to the exception class name when an inner step raises without a message, so the CLI never prints a bare `Setup failed:` (or `Connection failed:`) with nothing after the colon. (#127)
