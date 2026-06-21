"""Validate the in-repo Claude Code plugin scaffold (#191).

Asserts the plugin manifest + marketplace entry are valid JSON with the required
keys, their versions are internally consistent, and each shell-out skill exists
with frontmatter whose ``name`` matches its directory. Frontmatter is parsed
without a YAML dependency (kagura-memory's dev deps don't include PyYAML).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_JSON = REPO_ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE_JSON = REPO_ROOT / ".claude-plugin" / "marketplace.json"
SKILLS = ["doctor", "auth", "setup", "ingest", "resource", "files"]
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_plugin_json_valid_and_required_keys() -> None:
    assert PLUGIN_JSON.is_file(), f"missing {PLUGIN_JSON}"
    data = _load(PLUGIN_JSON)
    for key in ("name", "version", "description", "author", "license"):
        assert key in data, f"plugin.json missing '{key}'"
    assert data["name"] == "kagura-memory"
    assert data["license"] == "MIT"
    assert _SEMVER.match(data["version"]), data["version"]


def test_marketplace_json_valid_and_entry() -> None:
    assert MARKETPLACE_JSON.is_file(), f"missing {MARKETPLACE_JSON}"
    data = _load(MARKETPLACE_JSON)
    for key in ("name", "description", "owner", "plugins"):
        assert key in data, f"marketplace.json missing '{key}'"
    assert data["name"] == "kagura-memory"
    assert isinstance(data["plugins"], list) and len(data["plugins"]) == 1
    entry = data["plugins"][0]
    assert entry["name"] == "kagura-memory"
    assert entry["source"] == "./"


def test_plugin_and_marketplace_versions_synced() -> None:
    # Internal consistency between the two manifests. Keeping these in sync with
    # kagura_memory.__version__ at release time is a documented follow-up (the
    # version lives only in __init__.py per .claude/rules/versioning.md, so
    # coupling here would require wiring /release — see the PR).
    assert _load(PLUGIN_JSON)["version"] == _load(MARKETPLACE_JSON)["plugins"][0]["version"]


def _frontmatter(text: str) -> dict[str, str]:
    """Parse simple ``key: value`` SKILL.md frontmatter without PyYAML."""
    assert text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    _, fm, _body = text.split("---\n", 2)
    out: dict[str, str] = {}
    for line in fm.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


@pytest.mark.parametrize("name", SKILLS)
def test_skill_md_exists_with_matching_frontmatter(name: str) -> None:
    path = REPO_ROOT / "skills" / name / "SKILL.md"
    assert path.is_file(), f"missing {path}"
    text = path.read_text(encoding="utf-8")
    front = _frontmatter(text)
    assert front.get("name") == name, f"{name}/SKILL.md frontmatter name must equal its dir"
    assert front.get("description"), f"{name}/SKILL.md needs a description"
    body = text.split("---\n", 2)[2]
    assert body.strip(), f"{name}/SKILL.md needs a body"
