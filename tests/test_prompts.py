"""Cover prompts._format_session_content's artifact-formatting branch.

The message path was already exercised via the agent tests, but the artifact
branch (metadata lines, preview truncation, and the MAX_ARTIFACTS cap) had no
coverage because no test passed a session carrying artifacts.
"""

from __future__ import annotations

from kagura_memory.models import Artifact, Message, Session
from kagura_memory.prompts import (
    ARTIFACT_PREVIEW_LENGTH,
    MAX_ARTIFACTS,
    _format_session_content,
)


def _session(artifacts: list[Artifact]) -> Session:
    return Session(messages=[Message(role="user", content="hi")], artifacts=artifacts)


def test_no_artifacts_yields_empty_artifact_text() -> None:
    messages, artifacts = _format_session_content(_session([]))
    assert "[USER]: hi" in messages
    assert artifacts == ""


def test_artifact_metadata_is_rendered() -> None:
    art = Artifact(type="code", content="print('x')", source="main.py", language="python")
    _messages, artifacts = _format_session_content(_session([art]))
    assert "## Attached Artifacts:" in artifacts
    assert "[1] Type: code" in artifacts
    assert "Source: main.py" in artifacts
    assert "Language: python" in artifacts
    assert "print('x')" in artifacts


def test_optional_metadata_omitted_and_long_content_truncated() -> None:
    long = "a" * (ARTIFACT_PREVIEW_LENGTH + 50)
    _messages, artifacts = _format_session_content(
        _session([Artifact(type="document", content=long)])
    )
    # preview is the first ARTIFACT_PREVIEW_LENGTH chars + ellipsis, not the full body
    assert "a" * ARTIFACT_PREVIEW_LENGTH + "..." in artifacts
    assert long not in artifacts
    # source/language are None here, so their lines must not appear
    assert "Source:" not in artifacts
    assert "Language:" not in artifacts


def test_artifacts_are_capped_at_max() -> None:
    arts = [Artifact(type="code", content=f"c{i}") for i in range(MAX_ARTIFACTS + 3)]
    _messages, artifacts = _format_session_content(_session(arts))
    assert f"[{MAX_ARTIFACTS}] Type:" in artifacts
    assert f"[{MAX_ARTIFACTS + 1}] Type:" not in artifacts
