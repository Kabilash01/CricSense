"""Configuration: YAML in configs/ -> typed, frozen dataclasses.

All paths used anywhere in the package come from here. Relative paths in the YAML are
resolved against the repo root, so a clean clone runs without editing anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT_ENV_VAR = "CRICSENSE_ROOT"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    player_weights: Path
    ball_weights: Path
    video_dir: Path
    output_dir: Path
    calibration_dir: Path


@dataclass(frozen=True)
class VideoOptions:
    resize: tuple[int, int] | None


@dataclass(frozen=True)
class CameraOptions:
    match_scale: float
    orb_features: int
    ransac_px: float
    min_inliers: int
    max_chain_frames: int
    smooth_window: int
    outlier_px: float
    overlay_mask: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class Config:
    root: Path
    device: str
    paths: Paths
    video: VideoOptions
    camera: CameraOptions
    players: dict[str, Any]  # typed in Phase 2
    ball: dict[str, Any]  # typed in Phase 3
    commentary: dict[str, Any] = field(default_factory=dict)

    def output_dir_for(self, video_id: str) -> Path:
        return self.paths.output_dir / video_id

    def calibration_path(self, video_id: str) -> Path:
        return self.paths.calibration_dir / f"{video_id}.yaml"


def find_root(start: Path | None = None) -> Path:
    """Repo root: $CRICSENSE_ROOT, else the nearest ancestor of `start` holding configs/default.yaml."""
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        root = Path(env).expanduser().resolve()
        if not (root / "configs" / "default.yaml").is_file():
            raise ConfigError(f"${ROOT_ENV_VAR}={env} has no configs/default.yaml")
        return root
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "configs" / "default.yaml").is_file():
            return candidate
    raise ConfigError(
        f"no configs/default.yaml found in {here} or its parents; "
        f"run from inside the repo or set ${ROOT_ENV_VAR}"
    )


def video_id_from_path(video: Path | str) -> str:
    return Path(video).stem


def load_config(root: Path | None = None) -> Config:
    root = find_root() if root is None else root.resolve()
    configs = root / "configs"
    default = _read_yaml(configs / "default.yaml")

    try:
        raw_paths = default["paths"]
        paths = Paths(**{name: _resolve(root, raw_paths[name]) for name in Paths.__dataclass_fields__})
    except KeyError as e:
        raise ConfigError(f"configs/default.yaml: missing paths.{e.args[0]}") from None

    resize = (default.get("video") or {}).get("resize")
    if resize is not None:
        if not (isinstance(resize, list) and len(resize) == 2 and all(isinstance(v, int) and v > 0 for v in resize)):
            raise ConfigError(f"configs/default.yaml: video.resize must be null or [width, height], got {resize!r}")
        resize = (resize[0], resize[1])

    return Config(
        root=root,
        device=str(default.get("device", "cpu")),
        paths=paths,
        video=VideoOptions(resize=resize),
        camera=_camera_options(default.get("camera") or {}),
        players=_read_yaml(configs / "players.yaml"),
        ball=_read_yaml(configs / "ball.yaml"),
        commentary=_read_yaml(configs / "commentary.yaml") if (configs / "commentary.yaml").is_file() else {},
    )


def _camera_options(raw: dict[str, Any]) -> CameraOptions:
    where = "configs/default.yaml: camera"
    try:
        opts = CameraOptions(
            match_scale=float(raw["match_scale"]),
            orb_features=int(raw["orb_features"]),
            ransac_px=float(raw["ransac_px"]),
            min_inliers=int(raw["min_inliers"]),
            max_chain_frames=int(raw["max_chain_frames"]),
            smooth_window=int(raw.get("smooth_window", 11)),
            outlier_px=float(raw.get("outlier_px", 6.0)),
            overlay_mask=tuple(tuple(float(v) for v in rect) for rect in raw.get("overlay_mask") or ()),
        )
    except KeyError as e:
        raise ConfigError(f"{where}.{e.args[0]} is missing") from None
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where}: {e}") from None
    if not 0 < opts.match_scale <= 1:
        raise ConfigError(f"{where}.match_scale must be in (0, 1], got {opts.match_scale}")
    if opts.orb_features < 100 or opts.min_inliers < 8 or opts.ransac_px <= 0 or opts.max_chain_frames < 0:
        raise ConfigError(
            f"{where}: orb_features >= 100, min_inliers >= 8, ransac_px > 0 and max_chain_frames >= 0 required"
        )
    for rect in opts.overlay_mask:
        if len(rect) != 4 or not (0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1):
            raise ConfigError(f"{where}.overlay_mask: bad rectangle {list(rect)}")
    return opts


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _resolve(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p
