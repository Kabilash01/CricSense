"""Synthetic broadcast footage with exact ground truth.

A top-down "world" image of the pitch area (textured grass, painted creases) is filmed by a
virtual camera behind the bowler that pans and zooms. Every frame's true pixel -> pitch
mapping is known, so calibration and camera tracking can be scored in metres.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cricsense.calibration import pitch_model

PX_PER_M = 60.0  # fine enough that zoomed-in frames still have texture to match
# World image covers x in [-10, 30] m along the pitch and y in [-15, 15] m across it.
WORLD_X0, WORLD_Y0 = -10.0, -15.0
WORLD_W, WORLD_H = int(40 * PX_PER_M), int(30 * PX_PER_M)
FRAME_SIZE = (960, 540)  # width, height


def pitch_to_world() -> np.ndarray:
    """Pitch metres -> world pixels. World u runs across the pitch, v along it."""
    return np.array([[0, PX_PER_M, -WORLD_Y0 * PX_PER_M], [PX_PER_M, 0, -WORLD_X0 * PX_PER_M], [0, 0, 1]], float)


def render_world(seed: int = 0, creases: bool = True) -> np.ndarray:
    """Top-down world image: multi-scale grass texture, optionally with the painted creases.

    Texture at several scales (like real grass, cracks and mowing stripes) keeps features
    stable under perspective resampling at every zoom level. creases=False gives footage
    of "somewhere else", for testing cuts.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((WORLD_H, WORLD_W))
    for k, cell in enumerate((4, 8, 16, 32, 64)):
        small = rng.random((WORLD_H // cell + 2, WORLD_W // cell + 2))
        acc += cv2.resize(small, None, fx=cell, fy=cell, interpolation=cv2.INTER_CUBIC)[:WORLD_H, :WORLD_W] * 0.6**k
    base = ((acc - acc.min()) / (acc.max() - acc.min()) * 255).astype(np.uint8)
    world = cv2.merge([base // 3, 60 + base // 2, base // 3])
    if creases:
        to_world = pitch_to_world()
        for a, b in pitch_model.crease_segments():
            pa = to_world @ np.array([*a, 1.0])
            pb = to_world @ np.array([*b, 1.0])
            cv2.line(world, (round(pa[0]), round(pa[1])), (round(pb[0]), round(pb[1])), (255, 255, 255), 4)
    return world


def camera_homography(pan_px: float = 0.0, tilt_px: float = 0.0, zoom: float = 1.0) -> np.ndarray:
    """World pixels -> frame pixels for a camera behind the bowler looking down the pitch.

    The base view is a fixed perspective; pan/tilt/zoom are applied in the image, which is
    exactly how a PTZ camera's view changes.
    """
    w, h = FRAME_SIZE
    to_world = pitch_to_world()
    # Base view: where four pitch points land in the frame (striker end far, bowler end near).
    pitch_quad = np.array([[-2, -6], [-2, 6], [24, -9], [24, 9]], float)
    frame_quad = np.array([[330, 80], [630, 80], [60, 520], [900, 520]], float)
    world_quad = np.c_[pitch_quad, np.ones(4)] @ to_world.T
    base = cv2.getPerspectiveTransform(world_quad[:, :2].astype(np.float32), frame_quad.astype(np.float32))
    cx, cy = w / 2, h / 2
    ptz = np.array([[zoom, 0, cx - zoom * cx + pan_px], [0, zoom, cy - zoom * cy + tilt_px], [0, 0, 1]])
    return ptz @ base


def true_image_to_pitch(world_to_frame: np.ndarray) -> np.ndarray:
    H = np.linalg.inv(world_to_frame @ pitch_to_world())
    return H / H[2, 2]


def render_frame(world: np.ndarray, world_to_frame: np.ndarray, score_bar: bool = True) -> np.ndarray:
    frame = cv2.warpPerspective(world, world_to_frame, FRAME_SIZE, borderValue=(40, 90, 40))
    if score_bar:  # a static on-screen graphic, like the broadcast score bar
        w, h = FRAME_SIZE
        cv2.rectangle(frame, (0, int(0.88 * h)), (w, h), (90, 20, 60), -1)
        for i in range(12):
            cv2.putText(frame, f"{i * 7 % 10}", (20 + i * 75, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homog = np.c_[pts, np.ones(len(pts))] @ H.T
    return homog[:, :2] / homog[:, 2:3]


def camera_path(n: int) -> list[tuple[float, float, float]]:
    """(pan, tilt, zoom) per frame: a delivery-like pan and a ~25% zoom in and back out."""
    s = np.linspace(0, 1, n)
    return [(60 * np.sin(np.pi * v), -40 * v, 1 + 0.25 * np.sin(np.pi * v)) for v in s]


def write_video(path: Path, frames: list[np.ndarray], fps: float = 25.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, FRAME_SIZE)
    assert writer.isOpened()
    for f in frames:
        writer.write(f)
    writer.release()
    return path
