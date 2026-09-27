"""Data contracts between passes (ARCHITECTURE.md §4).

Every pass validates its input and output against these models, which is what lets the
passes run independently. Models forbid unknown fields: a pass cannot quietly add a field
such as a km/h speed that downstream code would then trust.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0"

# Timestamps must equal frame_id / fps to within this many seconds.
_T_TOLERANCE_S = 1e-3


class Role(str, Enum):
    STRIKER = "striker"
    NON_STRIKER = "non_striker"
    BOWLER = "bowler"
    KEEPER = "keeper"
    FIELDER = "fielder"
    UMPIRE = "umpire"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    RELEASE = "RELEASE"
    BOUNCE = "BOUNCE"
    CONTACT = "CONTACT"
    BOUNDARY = "BOUNDARY"
    WICKET = "WICKET"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerFrame(_Strict):
    frame_id: NonNegativeInt
    t: NonNegativeFloat  # seconds
    sid: PositiveInt  # stable id
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in image pixels
    role: Role  # from pitch geometry, never from the detector class
    pitch_xy: tuple[float, float] | None = None  # foot point in metres; None if uncalibrated
    identity: str | None = None  # from the scoreboard, else None
    identity_conf: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("box")
    @classmethod
    def _box_ordered(cls, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"box must satisfy x1 < x2 and y1 < y2, got {box}")
        return box


class BallFrame(_Strict):
    frame_id: NonNegativeInt
    t: NonNegativeFloat
    uv: tuple[float, float]  # image coords, y increases downward
    observed: bool  # False = Kalman prediction only
    # Projecting through the pitch homography is only valid when the ball touches the ground,
    # so this is set at the bounce frame and None everywhere else.
    pitch_xy: tuple[float, float] | None = None


class Event(_Strict):
    frame_id: NonNegativeInt
    t: NonNegativeFloat
    type: EventType
    actors: dict[Role, PositiveInt] = Field(default_factory=dict)  # role -> sid
    payload: dict[str, Any] = Field(default_factory=dict)  # length class, line offset, etc.
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1)  # which module emitted it


class VideoMeta(_Strict):
    video_id: str = Field(min_length=1)
    fps: PositiveFloat
    n_frames: NonNegativeInt
    width: PositiveInt
    height: PositiveInt


class RunInfo(_Strict):
    """How the pass ran; read by scripts/benchmark.py."""

    wall_time_s: NonNegativeFloat
    frames_processed: NonNegativeInt
    device: str


class _PassDoc(_Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    video: VideoMeta
    run: RunInfo | None = None

    def _check_timing(self, items: list[PlayerFrame] | list[BallFrame] | list[Event]) -> None:
        fps, n_frames = self.video.fps, self.video.n_frames
        for item in items:
            if item.frame_id >= n_frames:
                raise ValueError(f"frame_id {item.frame_id} >= n_frames {n_frames}")
            expected = item.frame_id / fps
            if abs(item.t - expected) > _T_TOLERANCE_S:
                raise ValueError(
                    f"frame_id {item.frame_id}: t={item.t} but frame_id/fps={expected:.4f}; "
                    "t must come from the source frame index"
                )


class PlayersDoc(_PassDoc):
    pass_name: Literal["players"] = "players"
    frames: list[PlayerFrame]

    @model_validator(mode="after")
    def _timing(self) -> PlayersDoc:
        self._check_timing(self.frames)
        return self


class BallDoc(_PassDoc):
    pass_name: Literal["ball"] = "ball"
    frames: list[BallFrame]

    @model_validator(mode="after")
    def _timing(self) -> BallDoc:
        self._check_timing(self.frames)
        return self


class EventsDoc(_PassDoc):
    pass_name: Literal["fuse"] = "fuse"
    events: list[Event]

    @model_validator(mode="after")
    def _timing(self) -> EventsDoc:
        self._check_timing(self.events)
        return self


# --- Calibration -------------------------------------------------------------------------

Matrix3 = tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]


class LandmarkObservation(_Strict):
    name: str = Field(min_length=1)  # a key of calibration.pitch_model.LANDMARKS
    image: tuple[float, float]  # pixels in the reference frame
    pitch: tuple[float, float]  # metres


class ReferenceView(_Strict):
    """One camera view, calibrated by clicking landmarks on one of its frames."""

    name: str = Field(min_length=1)
    frame_id: NonNegativeInt
    image_to_pitch: Matrix3
    fit_points: list[LandmarkObservation] = Field(min_length=4)
    check_points: list[LandmarkObservation] = Field(default_factory=list)
    fit_rms_px: NonNegativeFloat  # ~0 by construction with exactly 4 points
    check_rms_m: NonNegativeFloat | None = None  # held-out error: the real quality measure
    check_max_m: NonNegativeFloat | None = None


class CalibrationDoc(_Strict):
    """configs/calibration/<video_id>.yaml, written by `cricsense calibrate`."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    video_id: str = Field(min_length=1)
    image_size: tuple[PositiveInt, PositiveInt]  # width, height the views were clicked at
    views: list[ReferenceView] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_view_names(self) -> CalibrationDoc:
        names = [v.name for v in self.views]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate view names: {names}")
        return self

    def view(self, name: str) -> ReferenceView:
        for v in self.views:
            if v.name == name:
                return v
        raise KeyError(name)


class CameraFrame(_Strict):
    """Where one frame sits relative to the calibrated views."""

    frame_id: NonNegativeInt
    t: NonNegativeFloat
    view: str | None  # None: no reference view matched (cut, replay, close-up, other end)
    image_to_pitch: Matrix3 | None
    inliers: NonNegativeInt  # feature matches supporting the estimate

    @model_validator(mode="after")
    def _view_and_matrix_together(self) -> CameraFrame:
        if (self.view is None) != (self.image_to_pitch is None):
            raise ValueError("view and image_to_pitch must both be set or both be None")
        return self


class CameraDoc(_PassDoc):
    """outputs/<video_id>/camera.json: a pitch homography for every calibrated frame."""

    pass_name: Literal["camera"] = "camera"
    frames: list[CameraFrame]

    @model_validator(mode="after")
    def _timing(self) -> CameraDoc:
        self._check_timing(self.frames)
        return self
