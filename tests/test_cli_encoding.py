"""Regression tests for cp932/locale-independent CLI encoding (issue #197).

On Japanese/Chinese Windows the OS default text codec is cp932/cp950/gbk, so
file I/O that omits encoding=utf-8 decodes/encodes wrongly or crashes with
UnicodeDecodeError. These tests pin the JSON config I/O to UTF-8 and are written
to be red before the fix and green after, independent of the host locale.
"""

import json
from pathlib import Path

from kagura_memory.config import load_config
from kagura_memory.setup_claude import _read_json_safe, _write_json

_JP = "日本語"  # Japanese text (multibyte UTF-8)
# Lone high bytes: invalid UTF-8, so read_text(encoding=utf-8) raises UnicodeDecodeError.
_BAD_UTF8 = bytes([0x92, 0x93, 0xFF]) + b" not valid utf-8"


def _spy_encoding(monkeypatch, method_name):
    """Patch Path.<method_name> to record the encoding kwarg it received."""
    captured = {}
    original = getattr(Path, method_name)

    def spy(self, *args, **kwargs):
        captured["encoding"] = kwargs.get("encoding")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, method_name, spy)
    return captured


class TestSetupClaudeJsonEncoding:
    def test_read_json_safe_falls_back_on_undecodable_bytes(self, tmp_path: Path) -> None:
        """The 0x92 byte from the issue report is invalid UTF-8 as a lone byte.

        Before the fix _read_json_safe caught only FileNotFoundError /
        JSONDecodeError, so the UnicodeDecodeError propagated and crashed setup.
        """
        path = tmp_path / "settings.json"
        path.write_bytes(_BAD_UTF8)
        assert _read_json_safe(path) == {}

    def test_read_json_safe_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _read_json_safe(tmp_path / "nope.json") == {}

    def test_read_json_safe_decodes_multibyte_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        data = dict(note=_JP)
        path.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        assert _read_json_safe(path) == data

    def test_read_json_safe_uses_utf8_encoding(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps(dict()), encoding="utf-8")
        captured = _spy_encoding(monkeypatch, "read_text")
        _read_json_safe(path)
        assert captured["encoding"] == "utf-8"

    def test_write_json_uses_utf8_encoding(self, tmp_path: Path, monkeypatch) -> None:
        path = tmp_path / "cfg.json"
        captured = _spy_encoding(monkeypatch, "write_text")
        _write_json(path, dict(note=_JP))
        assert captured["encoding"] == "utf-8"

    def test_write_json_round_trips_through_read(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.json"
        data = dict(note=_JP, n=1)
        _write_json(path, data)
        assert _read_json_safe(path) == data


class TestLoadConfigEncoding:
    def test_load_config_reads_local_as_utf8(self, tmp_path: Path, monkeypatch) -> None:
        for key in ("KAGURA_API_KEY", "KAGURA_MCP_URL", "KAGURA_MODEL", "KAGURA_CONTEXT_ID"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(tmp_path)
        data = dict(api_key="k", model=_JP)
        (tmp_path / ".kagura.json").write_bytes(
            json.dumps(data, ensure_ascii=False).encode("utf-8")
        )
        captured = _spy_encoding(monkeypatch, "read_text")
        result = load_config()
        assert result["model"] == _JP
        assert captured["encoding"] == "utf-8"
