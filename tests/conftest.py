from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CLIP_FPS = 25.0
CLIP_FRAMES = 40
CLIP_SIZE = (160, 120)  # width, height


def frame_level(frame_id: int) -> int:
    """Gray level painted into synthetic frame `frame_id`, so a reader's order can be checked."""
    return (frame_id * 6) % 240 + 8


@pytest.fixture(scope="session")
def synthetic_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 40-frame, 25 fps clip whose frame i is a flat gray of frame_level(i)."""
    path = tmp_path_factory.mktemp("video") / "synthetic_clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), CLIP_FPS, CLIP_SIZE)
    assert writer.isOpened(), "OpenCV could not open a video writer"
    w, h = CLIP_SIZE
    for i in range(CLIP_FRAMES):
        writer.write(np.full((h, w, 3), frame_level(i), dtype=np.uint8))
    writer.release()
    return path
