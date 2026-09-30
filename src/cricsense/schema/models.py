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
    CONTACT = "CONTACT"  # bat on ball
    RUNS = "RUNS"  # the ball's outcome: runs off the bat and extras
    BOUNDARY = "BOUNDARY"  # payload kind: FOUR | SIX
    WICKET = "WICKET"
    WIDE = "WIDE"
    NO_BALL = "NO_BALL"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerFrame(_Strict):
    frame_id: NonNegativeInt
    t: NonNegativeFloat  # seconds
    sid: PositiveInt  # stable id
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in image pixels
    role: Role  # from pitch geometry, never from the detector class
    pitch_xy: tuple[float, float] | None = None  # foot point in metres; None if uncalibrated
    identity: str | None = None  # name, where one is known
    identity_conf: float = Field(default=0.0, ge=0.0, le=1.0)
    # "scoreboard": read off the broadcast bar (striker, non-striker, bowler);
    # "match_sheet": the fielding side's keeper from configs/match.yaml
    identity_source: Literal["scoreboard", "match_sheet"] | None = None
    team: str | None = None  # batting side for batters, fielding side for the rest; None for umpires

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
    observed: bool  # True: a detection; False: interpolated between detections
    delivery_id: NonNegativeInt | None = None
    # Projecting through the pitch homography is only valid when the ball touches the ground,
    # so this is set at the bounce frame and None everywhere else.
    pitch_xy: tuple[float, float] | None = None


class LengthClass(str, Enum):
    YORKER = "yorker"
    FULL = "full"
    GOOD = "good"
    SHORT = "short"


class Delivery(_Strict):
    """One ball bowled, as seen from the main camera."""

    delivery_id: NonNegativeInt
    shot_frames: tuple[NonNegativeInt, NonNegativeInt]  # first and last frame of the camera shot
    release_frame: NonNegativeInt
    release_t: NonNegativeFloat
    release_point: tuple[float, float, float]  # metres; on the release sight line at hand height
    detections: NonNegativeInt
    bounce_t: NonNegativeFloat | None = None
    bounce_xy: tuple[float, float] | None = None  # metres; the ball is on the ground, so this is measured
    bounce_confirmed: bool = False  # False: the ball was not seen after pitching
    length_m: float | None = None  # distance of the bounce from the striker's stumps
    line_m: float | None = None  # sideways offset from the middle-stump line (+ to image right)
    length_class: LengthClass | None = None
    # the broadcaster's radar reading shown after this ball: the authoritative speed
    radar_speed_kph: float | None = Field(default=None, gt=0)
    # our own estimate from release and bounce timing; single-camera, see ball/physics.py
    speed_estimate_kph: float | None = Field(default=None, gt=0)
    track_model: str | None = None  # weights file whose detections gave the track


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
    deliveries: list[Delivery] = Field(default_factory=list)

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


# --- Scoreboard --------------------------------------------------------------------------


class BatterLine(_Strict):
    name: str | None
    runs: NonNegativeInt | None
    balls: NonNegativeInt | None
    on_strike: bool


class ScoreboardReading(_Strict):
    """What the broadcast score bar showed on one frame. Evidence, never authoritative."""

    frame_id: NonNegativeInt
    t: NonNegativeFloat
    batters: list[BatterLine] = Field(min_length=2, max_length=2)
    runs: NonNegativeInt | None
    wickets: int | None = Field(default=None, ge=0, le=10)
    legal_balls: NonNegativeInt | None  # overs "11.1" -> 67
    bowler: str | None
    bowler_wickets: int | None = Field(default=None, ge=0, le=10)
    bowler_runs: NonNegativeInt | None
    bowler_legal_balls: NonNegativeInt | None
    speed_kph: float | None = Field(default=None, gt=0)  # the broadcaster's radar, as displayed
    raw: dict[str, str] = Field(default_factory=dict)  # OCR text per field, for auditing


class ScoreboardDoc(_PassDoc):
    """outputs/<video_id>/scoreboard.json: readings on frames where the score bar was visible."""

    pass_name: Literal["scoreboard"] = "scoreboard"
    layout: str
    readings: list[ScoreboardReading]

    @model_validator(mode="after")
    def _timing(self) -> ScoreboardDoc:
        self._check_timing(self.readings)
        return self


# --- Calibration -------------------------------------------------------------------------

Matrix3 = tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]
Matrix34 = tuple[
    tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]
]


class LandmarkObservation(_Strict):
    name: str = Field(min_length=1)  # a key of calibration.pitch_model.LANDMARKS
    image: tuple[float, float]  # pixels in the reference frame
    pitch: tuple[float, float]  # metres


class ReferenceView(_Strict):
    """One camera view, calibrated by clicking landmarks on one of its frames."""

    name: str = Field(min_length=1)
    # The video `frame_id` belongs to, relative to the repo root. Broadcast cameras stay put
    # for a whole match, so one clicked view serves every clip of it. None: this video.
    source_video: str | None = None
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
    image_to_pitch: Matrix3 | None  # ground plane (z = 0): pixels -> pitch metres
    # full 3D camera: pitch metres (x, y, z up) -> homogeneous pixels; places things above the ground
    projection: Matrix34 | None = None
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


class BallState(_Strict):
    """The score engine's match state after one delivery, and how it squares with the score bar."""

    delivery_id: NonNegativeInt
    frame_id: NonNegativeInt  # when the state holds: the ball's outcome, or its release if unknown
    t: NonNegativeFloat
    over: str  # "11.5": completed overs . balls
    runs: NonNegativeInt
    wickets: int = Field(ge=0, le=10)
    legal_balls: NonNegativeInt
    striker: str | None
    non_striker: str | None
    bowler: str | None
    batters: dict[str, tuple[int, int]]  # name -> (runs, balls faced)
    bowlers: dict[str, tuple[int, int, int]]  # name -> (runs conceded, legal balls, wickets)
    # "engine": updated by the laws from this ball's outcome; "resync": the engine had no evidence
    # for a ball before this one and took the score bar's state; "no_evidence": unchanged
    source: Literal["engine", "resync", "no_evidence"]
    checks: dict[str, bool] = Field(default_factory=dict)  # engine vs score bar, check by check
    notes: list[str] = Field(default_factory=list)


class MatchDoc(_PassDoc):
    """outputs/<video_id>/match.json: the score engine's ball-by-ball state."""

    pass_name: Literal["score"] = "score"
    states: list[BallState]

    @model_validator(mode="after")
    def _timing(self) -> MatchDoc:
        self._check_timing(self.states)
        return self


class CommentaryLine(_Strict):
    delivery_id: NonNegativeInt
    frame_id: NonNegativeInt  # when the line starts
    t: NonNegativeFloat
    kind: Literal["ball", "after"]  # as the ball is played / after it (score, over, milestone)
    text: str = Field(min_length=1)
    source: Literal["template", "llm"]
    fallback_reason: str | None = None  # why an LLM line was replaced by the template
    max_s: PositiveFloat  # time available before the next line or ball


class CommentaryDoc(_PassDoc):
    """outputs/<video_id>/commentary.json: lines, and the fact sheets they were written from."""

    pass_name: Literal["commentary"] = "commentary"
    engine: str  # "template" or "<provider>:<model>"
    lines: list[CommentaryLine]
    facts: list[dict[str, Any]]

    @model_validator(mode="after")
    def _timing(self) -> CommentaryDoc:
        self._check_timing(self.lines)
        return self


class PoseFrame(_Strict):
    """COCO-17 keypoints of one player in one frame: (x, y, confidence) in image pixels."""

    frame_id: NonNegativeInt
    t: NonNegativeFloat
    sid: PositiveInt
    role: Role
    keypoints: list[tuple[float, float, float]] = Field(min_length=17, max_length=17)


class BatterAction(_Strict):
    """What the striker's body did on one delivery, measured from pose on the calibrated pitch."""

    delivery_id: NonNegativeInt
    sid: PositiveInt | None
    footwork: Literal["front foot", "back foot", "in the crease"] | None  # None: feet not seen well enough
    stride_m: float | None  # + towards the bowler (front foot), - back towards the stumps
    swing: Literal["played", "left"] | None
    swing_peak_frame: NonNegativeInt | None
    hand_speed: float | None  # peak wrist speed, body heights per second
    frames: NonNegativeInt  # frames with the striker's pose


class PoseDoc(_PassDoc):
    """outputs/<video_id>/pose.json: striker keypoints around each delivery and what they show."""

    pass_name: Literal["pose"] = "pose"
    frames: list[PoseFrame]
    batters: list[BatterAction]

    @model_validator(mode="after")
    def _timing(self) -> PoseDoc:
        self._check_timing(self.frames)
        return self

