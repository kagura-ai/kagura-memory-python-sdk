---
paths:
  - "src/**"
  - "tests/**"
---

# Python Coding Rules

## Style
- snake_case for functions/variables, PascalCase for classes
- Type hints required on all function signatures
- Google-style docstrings on public functions
- No `print()` — use `VerboseLogger` for user output

## Async
- All HTTP calls use `httpx.AsyncClient` (async)
- Use `async with` for client lifecycle
- Never block the event loop

## Error Handling
- Use custom exceptions from `exceptions.py` (KaguraError hierarchy)
- Never catch broad `Exception` without re-raising or logging
- Auth errors (`KaguraAuthError`) must always surface to caller

## Security
- Never store API keys as instance attributes
- Never write secrets to `os.environ`
- Enforce HTTPS on MCP URLs (allow localhost for dev)

## Testing
- Use `pytest-asyncio` for async tests
- Mock `httpx.AsyncClient.post` with `AsyncMock`
- Test error paths (401, network failure, missing session)
- Always close agents/clients in tests (use `try/finally` or `async with`)
