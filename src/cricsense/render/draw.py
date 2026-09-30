"""Drawing primitives for the overlay: anti-aliased text chips and alpha blending.

Text is rendered once per distinct string with PIL (the DejaVu font that ships with
matplotlib, so every machine with the env renders the same) and cached as an RGBA patch;
per frame only a numpy blend runs.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Color = tuple[int, int, int]  # BGR, as OpenCV frames are


@lru_cache(maxsize=8)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    from matplotlib import font_manager

    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=4096)
def chip(
    text: str,
    size: int = 20,
    fg: Color = (255, 255, 255),
    bg: Color | None = (20, 20, 20),
    bold: bool = False,
    pad: tuple[int, int] = (8, 4),
    bg_alpha: int = 200,
    radius: int = 6,
) -> np.ndarray:
    """Text on an optional rounded background, as a BGRA patch (cached per argument set)."""
    f = font(size, bold)
    x0, y0, x1, y1 = f.getbbox(text)
    w, h = x1 - x0 + 2 * pad[0], (f.getmetrics()[0] + f.getmetrics()[1]) + 2 * pad[1]
    img = Image.new("RGBA", (max(w, 1), max(h, 1)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if bg is not None:
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=(bg[2], bg[1], bg[0], bg_alpha))
    d.text((pad[0] - x0, pad[1]), text, font=f, fill=(fg[2], fg[1], fg[0], 255))
    rgba = np.asarray(img)
    return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])  # RGBA -> BGRA


def blend(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> tuple[int, int, int, int]:
    """Alpha-blend a BGRA patch onto the frame at (x, y), clipped to the frame. Returns the box."""
    h, w = patch.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, fw), min(y + h, fh)
    if x0 >= x1 or y0 >= y1:
        return (x0, y0, x0, y0)
    p = patch[y0 - y : y1 - y, x0 - x : x1 - x]
    a = p[..., 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1]
    roi[:] = (p[..., :3] * a + roi * (1.0 - a)).astype(np.uint8)
    return (x0, y0, x1, y1)


def panel(width: int, height: int, color: Color = (20, 20, 20), alpha: int = 190, radius: int = 10) -> np.ndarray:
    return _panel(width, height, color, alpha, radius).copy()


@lru_cache(maxsize=64)
def _panel(width: int, height: int, color: Color, alpha: int, radius: int) -> np.ndarray:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=(color[2], color[1], color[0], alpha))
    return np.ascontiguousarray(np.asarray(img)[..., [2, 1, 0, 3]])


def paste(dst: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite a BGRA patch onto a BGRA canvas (for building panels)."""
    h, w = patch.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    p = patch[y0 - y : y1 - y, x0 - x : x1 - x].astype(np.float32)
    d = dst[y0:y1, x0:x1].astype(np.float32)
    a = p[..., 3:4] / 255.0
    out_a = a + d[..., 3:4] / 255.0 * (1 - a)
    rgb = (p[..., :3] * a + d[..., :3] * (d[..., 3:4] / 255.0) * (1 - a)) / np.maximum(out_a, 1e-6)
    dst[y0:y1, x0:x1, :3] = rgb.astype(np.uint8)
    dst[y0:y1, x0:x1, 3:4] = (out_a * 255).astype(np.uint8)
