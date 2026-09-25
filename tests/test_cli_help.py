"""Every command's ``--help`` keeps its examples one per line (#285)."""

import click
import pytest

from kagura_memory.cli import main


def _commands(cmd: click.Command, path: tuple[str, ...]):
    yield path, cmd
    if isinstance(cmd, click.Group):
        for name, sub in cmd.commands.items():
            yield from _commands(sub, (*path, name))


def _example_lines(help_text: str) -> list[str]:
    """The lines under ``Example:`` / ``Examples:`` up to the next blank line."""
    lines = help_text.splitlines()
    found: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() in ("Example:", "Examples:"):
            for example in lines[i + 1 :]:
                if not example.strip():
                    break
                found.append(example.strip())
    return found


_WITH_EXAMPLES = [
    (" ".join(path), cmd)
    for path, cmd in _commands(main, ("kagura",))
    if _example_lines(cmd.help or "")
]


def test_commands_with_examples_are_found():
    assert len(_WITH_EXAMPLES) > 40


@pytest.mark.parametrize("name,cmd", _WITH_EXAMPLES, ids=[n for n, _ in _WITH_EXAMPLES])
def test_examples_render_one_per_line(name, cmd):
    """Without ``\\b`` above them, click rewraps the examples into one run-on line."""
    with click.Context(cmd, info_name=name) as ctx:
        rendered = {line.strip() for line in cmd.get_help(ctx).splitlines()}
    missing = [line for line in _example_lines(cmd.help or "") if line not in rendered]
    assert not missing, f"{name} --help rewraps: {missing}"
