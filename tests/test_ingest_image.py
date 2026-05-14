"""Tests for ingest._image.preprocess (Pillow-based image preprocessing).

The preprocess pipeline is on the hot path for every Vision LLM call; bugs
here ship silently because the Vision flow is opt-in. These tests verify:

* The output is JPEG (re-encoded for size reduction).
* Long-edge resize to 1568 px is applied for oversized inputs.
* Small inputs pass through without upscaling.
* EXIF metadata is stripped (privacy + size).
* Decompression bombs are rejected as KaguraIngestError.
* Malformed bytes surface as KaguraIngestError, not ImagePIL exceptions.
"""

from __future__ import annotations

import io

import pytest

# Skip the whole module when Pillow isn't installed. This file exercises
# kagura_memory.ingest._image which depends on Pillow (an optional extra:
# `pip install kagura-memory[ingest]`). Without this guard, pytest fails
# at collection time on a bare `pip install kagura-memory[dev]` install.
pytest.importorskip("PIL", reason="Pillow not installed — install [ingest] extras")

from PIL import Image  # noqa: E402  type: ignore[import-not-found]
from PIL.ExifTags import TAGS  # noqa: E402  type: ignore[import-not-found]

from kagura_memory.exceptions import KaguraIngestError  # noqa: E402
from kagura_memory.ingest._image import preprocess  # noqa: E402


def _make_jpeg(width: int, height: int, color: str = "red") -> bytes:
    """Build a single-color JPEG of the requested size."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_jpeg_with_exif() -> bytes:
    """Build a JPEG with EXIF Make+Model tags for the strip test."""
    img = Image.new("RGB", (400, 400), color="blue")
    exif = img.getexif()
    exif[0x010F] = "TestMake"  # Make
    exif[0x0110] = "TestModel"  # Model
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_output_is_jpeg() -> None:
    body = _make_jpeg(800, 600)
    out, mime = preprocess(body)
    assert mime == "image/jpeg"
    # JPEG magic bytes — first 3 bytes are 0xFFD8FF.
    assert out[:3] == b"\xff\xd8\xff"


def test_output_is_decodable_image() -> None:
    body = _make_jpeg(800, 600)
    out, _ = preprocess(body)
    decoded = Image.open(io.BytesIO(out))
    decoded.load()  # forces decode; raises if bytes are invalid
    assert decoded.format == "JPEG"
    assert decoded.mode == "RGB"


def test_oversized_image_resized_to_long_edge_1568() -> None:
    """A 4000×3000 input (long edge 4000) must come out with long edge ≤ 1568."""
    body = _make_jpeg(4000, 3000)
    out, _ = preprocess(body)
    decoded = Image.open(io.BytesIO(out))
    w, h = decoded.size
    assert max(w, h) <= 1568
    # Aspect ratio preserved within reasonable rounding.
    in_ratio = 4000 / 3000
    out_ratio = w / h
    assert abs(in_ratio - out_ratio) < 0.02


def test_oversized_tall_image_resized_correctly() -> None:
    body = _make_jpeg(1000, 5000)  # long edge is height
    out, _ = preprocess(body)
    decoded = Image.open(io.BytesIO(out))
    w, h = decoded.size
    assert max(w, h) <= 1568
    assert h > w  # orientation preserved


def test_small_image_passes_through_without_upscaling() -> None:
    body = _make_jpeg(400, 300)
    out, _ = preprocess(body)
    decoded = Image.open(io.BytesIO(out))
    # Phase 1 only downsamples; small inputs keep their dimensions.
    assert decoded.size == (400, 300)


def test_exif_stripped_from_output() -> None:
    body = _make_jpeg_with_exif()
    # Sanity check: input has EXIF tags we just embedded.
    decoded_in = Image.open(io.BytesIO(body))
    in_exif_keys = {TAGS.get(k, k) for k in decoded_in.getexif().keys()}
    assert in_exif_keys & {"Make", "Model"}, (
        f"fixture should embed at least one EXIF tag, got: {in_exif_keys}"
    )

    out, _ = preprocess(body)
    decoded_out = Image.open(io.BytesIO(out))
    out_exif_keys = {TAGS.get(k, k) for k in decoded_out.getexif().keys()}
    assert not (out_exif_keys & {"Make", "Model"}), (
        f"EXIF Make/Model should be stripped, still present: {out_exif_keys}"
    )


def test_palette_mode_input_converted_to_rgb() -> None:
    """Palette-mode (P mode) inputs must be converted before encoding."""
    img = Image.new("P", (200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out, _ = preprocess(buf.getvalue())
    decoded = Image.open(io.BytesIO(out))
    assert decoded.mode == "RGB"


def test_rgba_input_converted_to_rgb() -> None:
    """RGBA inputs (PNG with transparency) must collapse to RGB for JPEG output."""
    img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out, _ = preprocess(buf.getvalue())
    decoded = Image.open(io.BytesIO(out))
    assert decoded.mode == "RGB"


def test_decompression_bomb_rejected() -> None:
    """Pillow raises DecompressionBombError; preprocess wraps as KaguraIngestError."""
    # Synthesize a "bomb" by pretending an image's stated dimensions
    # exceed MAX_IMAGE_PIXELS. We can't easily fabricate this in a unit test
    # without writing a malformed file, so we patch Image.open to raise
    # DecompressionBombError directly.
    from unittest.mock import patch

    with patch("PIL.Image.open", side_effect=Image.DecompressionBombError("too big")):
        with pytest.raises(KaguraIngestError, match="decompression bomb"):
            preprocess(b"any bytes")


def test_malformed_bytes_raises_ingest_error() -> None:
    with pytest.raises(KaguraIngestError, match="image preprocessing failed"):
        preprocess(b"not an image at all, just garbage")


def test_empty_bytes_raises_ingest_error() -> None:
    with pytest.raises(KaguraIngestError):
        preprocess(b"")


def test_output_smaller_than_random_high_res_input() -> None:
    """Sanity check: re-encoding a 4000×3000 JPEG produces a smaller payload."""
    body = _make_jpeg(4000, 3000, color="green")
    out, _ = preprocess(body)
    # Single-color JPEGs compress aggressively; the resized one must be
    # smaller than the input both due to lower resolution and quality 85.
    assert len(out) < len(body)
