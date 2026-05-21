# Changelog

See [GitHub Releases](https://github.com/kagura-ai/kagura-memory-python-sdk/releases) for all release notes.

## Unreleased

- `kagura auth login` now requests `memory:read memory:write` by default; pass `--read-only` for read-only scope. `--scope` still accepts a custom set. (#122)
- `kagura auth login` falls back to platform openers (`wslview`/`cmd.exe` on WSL, `open` on macOS, `xdg-open` on Linux) when `webbrowser.open()` doesn't actually launch a browser. (#122)
- Device-flow prompt now nudges the user to sign in to the Kagura web UI in the same browser first, so the consent page renders correctly. (#122)
