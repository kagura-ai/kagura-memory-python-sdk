"""Anti-rot smoke test: every example under examples/ must still import.

Importing an example executes its module-level ``from kagura_memory
import ...`` line and its function defs (the ``if __name__ ==
"__main__"`` guard keeps ``main()`` from running, so nothing hits the
network). This catches the common way examples go stale: an SDK export
gets renamed or removed and the example silently breaks.

Examples that need an optional extra (e.g. the PDF stack behind
``kagura-memory[ingest-pdf]``) are skipped when that extra is absent,
rather than failing — a missing optional dep is an environment fact,
not example rot.
"""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.py"))

# Optional-extra modules: a missing one means the extra isn't installed.
_OPTIONAL_MODULES = {"fitz", "pymupdf", "PIL"}


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_imports(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        if e.name in _OPTIONAL_MODULES:
            pytest.skip(f"optional dependency not installed: {e.name}")
        raise
