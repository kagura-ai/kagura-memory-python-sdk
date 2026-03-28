"""Tests for VerboseLogger."""

from io import StringIO

from rich.console import Console

from kagura_memory.logger import VerboseLogger


def _make_logger(level: int) -> tuple[VerboseLogger, StringIO]:
    """Create a logger with captured output."""
    buf = StringIO()
    console = Console(file=buf, no_color=True, width=200)
    return VerboseLogger(level=level, console=console), buf


def test_action_at_level_1():
    """action() should output at level 1+."""
    logger, buf = _make_logger(1)
    logger.action("Testing", "details here")
    output = buf.getvalue()
    assert "Testing" in output
    assert "details here" in output


def test_action_at_level_0():
    """action() should be silent at level 0."""
    logger, buf = _make_logger(0)
    logger.action("Should not appear")
    assert buf.getvalue() == ""


def test_action_without_details():
    """action() with no details should still output."""
    logger, buf = _make_logger(1)
    logger.action("Solo action")
    assert "Solo action" in buf.getvalue()


def test_detail_at_level_2():
    """detail() should output at level 2+."""
    logger, buf = _make_logger(2)
    logger.detail("key", "value")
    output = buf.getvalue()
    assert "key" in output
    assert "value" in output


def test_detail_silent_at_level_1():
    """detail() should be silent at level 1."""
    logger, buf = _make_logger(1)
    logger.detail("key", "value")
    assert buf.getvalue() == ""


def test_debug_with_dict():
    """debug() should format dict as JSON."""
    logger, buf = _make_logger(3)
    logger.debug("Test Data", {"key": "value"})
    output = buf.getvalue()
    assert "Test Data" in output
    assert "key" in output


def test_debug_with_string():
    """debug() should handle plain strings."""
    logger, buf = _make_logger(3)
    logger.debug("Title", "plain string data")
    assert "plain string data" in buf.getvalue()


def test_debug_truncates_long_data():
    """debug() should truncate data over 2000 chars."""
    logger, buf = _make_logger(3)
    logger.debug("Long", "x" * 3000)
    output = buf.getvalue()
    assert "truncated" in output


def test_debug_silent_at_level_2():
    """debug() should be silent at level 2."""
    logger, buf = _make_logger(2)
    logger.debug("Title", {"key": "value"})
    assert buf.getvalue() == ""


def test_success_message():
    """success() should output at level 1+."""
    logger, buf = _make_logger(1)
    logger.success("Done!")
    assert "Done!" in buf.getvalue()


def test_warning_message():
    """warning() should output at level 1+."""
    logger, buf = _make_logger(1)
    logger.warning("Watch out")
    assert "Watch out" in buf.getvalue()


def test_error_always_shown():
    """error() should output even at level 0."""
    logger, buf = _make_logger(0)
    logger.error("Something broke")
    assert "Something broke" in buf.getvalue()


def test_success_silent_at_level_0():
    """success() should be silent at level 0."""
    logger, buf = _make_logger(0)
    logger.success("Hidden")
    assert buf.getvalue() == ""
