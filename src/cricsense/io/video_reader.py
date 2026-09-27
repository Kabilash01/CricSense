"""Frame source for every pass.

Migrated from CricSense-AI src/ingest/video_reader.py. The legacy reader numbered frames by
how many it had *returned*, so with skip=2 every frame_id was half the real one. Here
frame_id is always the index in the source video and t = frame_id / fps.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cricsense.schema import VideoMeta


@dataclass(frozen=True)
class Frame:
    frame_id: int  # index in the source video
    t: float  # seconds, frame_id / fps
    image: np.ndarray  # BGR


class VideoReader:
    """Iterate frames [start, end) of a video, keeping every `skip`-th one.

    Use as a context manager, or call release() when done.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        skip: int = 1,
        start: int = 0,
        end: int | None = None,
        resize: tuple[int, int] | None = None,
    ) -> None:
        if skip < 1:
            raise ValueError(f"skip must be >= 1, got {skip}")
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open video: {self.path}")

        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0:
            self.release()
            raise RuntimeError(f"video reports no frame rate: {self.path}")
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.width, self.height = resize or native

        self.skip = skip
        self.start = max(0, start)
        self.end = self.n_frames if end is None else min(end, self.n_frames)
        self.resize = resize

    def meta(self, video_id: str) -> VideoMeta:
        return VideoMeta(
            video_id=video_id, fps=self.fps, n_frames=self.n_frames, width=self.width, height=self.height
        )

    def __iter__(self) -> Iterator[Frame]:
        if self._cap is None:
            raise RuntimeError("reader already released")
        # Frame-accurate positioning: decode from 0 rather than trusting CAP_PROP_POS_FRAMES,
        # which lands on the nearest keyframe for many codecs.
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_id in range(self.end):
            keep = frame_id >= self.start and (frame_id - self.start) % self.skip == 0
            if not keep:
                if not self._cap.grab():
                    return
                continue
            ok, image = self._cap.read()
            if not ok:
                return
            if self.resize is not None:
                image = cv2.resize(image, self.resize, interpolation=cv2.INTER_LINEAR)
            yield Frame(frame_id=frame_id, t=frame_id / self.fps, image=image)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
