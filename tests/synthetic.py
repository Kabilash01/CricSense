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

PX_PER_M = 40.0  # fine enough that zoomed-in frames still have texture to match
# World image covers x in [-30, 70] m along the pitch and y in [-35, 35] m across it: all a
# camera in the stand behind the bowler can see at the zooms the tests use.
WORLD_X0, WORLD_Y0 = -30.0, -35.0
WORLD_W, WORLD_H = int(70 * PX_PER_M), int(100 * PX_PER_M)
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
        # the pitch itself: a worn, pale-brown strip (what a real broadcast shows under the creases)
        corners = [to_world @ np.array([x, y, 1.0]) for x, y in ((-3, -1.6), (-3, 1.6), (23, 1.6), (23, -1.6))]
        strip = np.zeros(world.shape[:2], np.uint8)
        cv2.fillPoly(strip, [np.array([(round(p[0]), round(p[1])) for p in corners], np.int32)], 1)
        tan = cv2.merge([90 + base // 5, 140 + base // 5, 175 + base // 6])
        world[strip == 1] = tan[strip == 1]
        for a, b in pitch_model.crease_segments():
            pa = to_world @ np.array([*a, 1.0])
            pb = to_world @ np.array([*b, 1.0])
            cv2.line(world, (round(pa[0]), round(pa[1])), (round(pb[0]), round(pb[1])), (255, 255, 255), 4)
    return world


CAMERA_CENTRE = np.array([86.0, 0.0, 12.0])  # metres: as measured for the real broadcast camera
LOOK_AT = np.array([10.0, 0.0, 0.0])
BASE_FOCAL = 4400.0  # px at 960 wide: ~12.5 deg field of view, like the real camera zoomed out


def camera_projection(pan_px: float = 0.0, tilt_px: float = 0.0, zoom: float = 1.0) -> np.ndarray:
    """3x4 projection of a real pan-tilt-zoom camera: fixed position, rotating and zooming.

    pan_px / tilt_px turn the camera by the angle that moves the image centre by that many
    pixels at the base zoom.
    """
    w, h = FRAME_SIZE
    fwd = LOOK_AT - CAMERA_CENTRE
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R0 = np.stack([right, down, fwd])  # world -> camera (x right, y down, z forward)
    pan, tilt = np.arctan2(pan_px, BASE_FOCAL), np.arctan2(tilt_px, BASE_FOCAL)
    Ry = np.array([[np.cos(pan), 0, -np.sin(pan)], [0, 1, 0], [np.sin(pan), 0, np.cos(pan)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(tilt), np.sin(tilt)], [0, -np.sin(tilt), np.cos(tilt)]])
    R = Rx @ Ry @ R0
    f = BASE_FOCAL * zoom
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ CAMERA_CENTRE]


def camera_homography(pan_px: float = 0.0, tilt_px: float = 0.0, zoom: float = 1.0) -> np.ndarray:
    """World pixels -> frame pixels for that camera (the ground plane only)."""
    ground_to_image = camera_projection(pan_px, tilt_px, zoom)[:, [0, 1, 3]]
    return ground_to_image @ np.linalg.inv(pitch_to_world())


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
