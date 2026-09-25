"""The guardrail export block in an always-loaded file such as ``AGENTS.md`` (#253).

``kagura guardrails digest --out`` and ``kagura setup codex|hermes|openclaw
--agents-md`` (#260) both put memory-cloud's export block
(``GET /api/v1/memory/guardrails/digest``) into a file here, so the two can
never disagree about what they replace.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
from pathlib import Path

# The export block's marker lines, matched at line start. Mirrors the server's
# Codex cloud recipe (memory-cloud docs/mcp-clients.md § Codex cloud), so a
# file written by either tool is maintained by the other.
_GUARDRAIL_BEGIN_RE = re.compile(r"^<!-- kagura-memory:guardrails begin[^\n]*$", re.M)
_GUARDRAIL_END_RE = re.compile(r"^<!-- kagura-memory:guardrails end -->$", re.M)
_GUARDRAIL_SPAN_RE = re.compile(
    _GUARDRAIL_BEGIN_RE.pattern + r".*?" + _GUARDRAIL_END_RE.pattern + r"\n?", re.M | re.S
)


def has_guardrail_block(text: str) -> bool:
    """True when ``text`` has a guardrail export begin marker line.

    Args:
        text: A file's content.

    Returns:
        Whether an earlier export block (complete or not) is in it.
    """
    return _GUARDRAIL_BEGIN_RE.search(text) is not None


def splice_guardrail_block(text: str, block: str) -> str:
    """Return ``text`` with the guardrail export ``block`` put in place.

    The block replaces an earlier one in place, or is appended after a blank
    line. An empty ``block`` (the context has no tool guardrails) removes an
    earlier block and the blank line before it, and never creates one — a
    file never keeps a guardrail the server no longer serves.

    Raises:
        ValueError: The fetched block does not have exactly one begin and one
            end marker line, begin first, or ``text`` holds more than one block
            or a broken one — unterminated, or its end before its begin (fix
            that by hand rather than guess).
    """
    block = block.strip()
    if block and (
        len(_GUARDRAIL_BEGIN_RE.findall(block)) != 1
        or len(_GUARDRAIL_END_RE.findall(block)) != 1
        or not _GUARDRAIL_SPAN_RE.search(block)
    ):
        # Out of order, the block would be written, and the next run would
        # refuse the file as broken (#285).
        raise ValueError(
            "fetched block does not have exactly one begin and one end marker line, in that order"
        )
    begins = len(_GUARDRAIL_BEGIN_RE.findall(text))
    span = _GUARDRAIL_SPAN_RE.search(text)
    if begins > 1 or begins != len(_GUARDRAIL_END_RE.findall(text)) or (begins and not span):
        raise ValueError(
            "the file has more than one guardrail block, or a broken one; fix it by hand"
        )
    if span:
        start, end = span.span()
        if not block:
            # Drop the blank line that separated the block from the text
            # above. A lone newline ends the line above (the block may sit
            # right under the user's own heading), so it stays.
            if text[:start].endswith("\n\n"):
                start -= 1
            return text[:start] + text[end:]
        return text[:start] + block + "\n" + text[end:]
    if not block:
        return text
    if not text:
        return block + "\n"
    return text + ("" if text.endswith("\n") else "\n") + "\n" + block + "\n"


def write_guardrail_block(path: Path, block: str) -> str:
    """Splice ``block`` into the file at ``path``; return what happened.

    ``"unchanged"`` when the file already carries this block — the begin
    marker embeds ``tool_triggered_version``, so an unchanged guardrail set
    rewrites nothing — ``"removed"`` when an empty digest dropped an earlier
    block, else ``"written"``. A symlink (``AGENTS.md`` -> ``CLAUDE.md``) is
    followed so the link survives, and an existing file is replaced
    atomically with its permission bits kept.

    Line endings are the file's own, on every platform: a file with any CRLF
    is written back with CRLF, anything else (and a new file) with LF, so a
    write changes the block and not every line of the file.
    """
    target = Path(os.path.realpath(path))
    exists = target.exists()
    raw = target.read_bytes().decode("utf-8") if exists else ""
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    new = splice_guardrail_block(text, block)
    if new == text:
        return "unchanged"
    if not exists:
        target.write_text(new, encoding="utf-8", newline=newline)
    else:
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as f:
                f.write(new)
            shutil.copymode(target, tmp)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    return "written" if block.strip() else "removed"
