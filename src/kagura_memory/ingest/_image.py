"""Image preprocessing for vision-LLM payload reduction.

Per design doc §8.3, every image sent to a Vision LLM is preprocessed:

* Resize so the long edge ≤ 1568 px (matches Claude vision recommendation,
  works well for Gemini too).
* Strip EXIF (location, device, timestamp) — privacy + non-essential bytes.
* Re-encode to JPEG quality 85 — smaller payload, no perceptible quality
  loss for OCR.

Pillow (``pillow>=10.4``) is a lazy import; the failure mode points the
user to the ``[ingest]`` extra.
"""

from __future__ import annotations

import io
import warnings
from typing import Any

from ..exceptions import KaguraIngestError

_LONG_EDGE_PX = 1568
_JPEG_QUALITY = 85
_MAX_IMAGE_PIXELS = 50_000_000  # tighter than Pillow's default ~89.5M

_pillow_initialized = False


def _load_pillow() -> Any:
    global _pillow_initialized
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as e:
        raise KaguraIngestError(
            "pillow is not installed. Install with: pip install 'kagura-memory[ingest]'"
        ) from e
    # Set the decompression-bomb cap exactly once on first load. Subsequent
    # calls leave Pillow's MAX_IMAGE_PIXELS alone — important when this
    # process shares Pillow with another library that has its own (possibly
    # stricter) policy.
    if not _pillow_initialized:
        if Image.MAX_IMAGE_PIXELS is None or Image.MAX_IMAGE_PIXELS > _MAX_IMAGE_PIXELS:
            Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
        # Pillow's default behavior is to emit a DecompressionBombWarning
        # between MAX_IMAGE_PIXELS and 2 * MAX_IMAGE_PIXELS and only raise
        # DecompressionBombError above 2 *. Converting the warning to an
        # exception makes the cap a hard limit — important for SSRF /
        # malicious-upload defense.
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        _pillow_initialized = True
    return Image


def preprocess(image_bytes: bytes) -> tuple[bytes, str]:
    """Resize, EXIF-strip, and re-encode an image.

    Args:
        image_bytes: Raw image bytes in any format Pillow accepts.

    Returns:
        ``(preprocessed_bytes, "image/jpeg")``. The MIME is always JPEG
        because we re-encode for size reduction.

    Raises:
        KaguraIngestError: If the image can't be decoded (corrupt, too
            large per ``MAX_IMAGE_PIXELS``, etc.).
    """
    Image = _load_pillow()  # noqa: N806 - Pillow's class is named Image; mirroring it preserves clarity
    try:
        # Decode in a context manager so the original (possibly RGBA / palette)
        # buffer is released as soon as we have the converted RGB copy.
        # Holding both buffers concurrently doubles peak RAM for high-MP inputs.
        with Image.open(io.BytesIO(image_bytes)) as src:
            rgb = src.convert("RGB")
        w, h = rgb.size
        scale = _LONG_EDGE_PX / max(w, h)
        if scale < 1.0:
            rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        rgb.close()
        return buf.getvalue(), "image/jpeg"
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as e:
        # DecompressionBombWarning fires between MAX and 2*MAX pixels (raised
        # as an exception via the simplefilter promotion in _load_pillow);
        # DecompressionBombError fires above 2*MAX (raised by Pillow directly).
        # They are in separate class hierarchies (Warning vs Exception), so
        # we must catch both explicitly.
        raise KaguraIngestError(f"image rejected as decompression bomb: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise KaguraIngestError(f"image preprocessing failed: {e}") from e
