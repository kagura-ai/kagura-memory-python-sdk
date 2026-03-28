Run all quality checks.

Steps:
1. Lint:
   ```
   uv run ruff check src/ tests/
   ```
2. Format check:
   ```
   uv run ruff format --check src/ tests/
   ```
3. Type check:
   ```
   uv run pyright src/
   ```
4. Tests:
   ```
   uv run pytest tests/ -v
   ```

Report any issues found. If fixable, ask to auto-fix (`ruff check --fix`, `ruff format`).
