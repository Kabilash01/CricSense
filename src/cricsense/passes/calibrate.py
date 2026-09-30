"""`cricsense calibrate`: reference-view calibration plus per-frame camera tracking.

Writes
    configs/calibration/<video_id>.yaml   the clicked reference view(s), human-readable
    outputs/<video_id>/calibration_check.png   the projected pitch over the reference frame
    outputs/<video_id>/camera.json   a pitch homography for every frame (or None)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cricsense.calibration import pitch_model
from cricsense.calibration.auto import CORNER_NAMES, find_views
from cricsense.calibration.camera_motion import read_frames, track_video
from cricsense.calibration.homography import CalibrationError, reprojection_errors
from cricsense.calibration.picker import PickerSession, draw_session, run_picker
from cricsense.config import Config, video_id_from_path
from cricsense.io import VideoReader, read_yaml_doc, write_doc, write_yaml_doc
from cricsense.schema import CalibrationDoc, CameraDoc, LandmarkObservation, ReferenceView

log = logging.getLogger("cricsense.calibrate")


@dataclass(frozen=True)
class CalibrateResult:
    calibration_path: Path
    check_image_path: Path | None
    camera_path: Path | None
    view: ReferenceView | None
    camera: CameraDoc | None


def parse_points(spec: str) -> dict[str, tuple[float, float]]:
    """'name=x,y;name=x,y' -> {name: (x, y)}."""
    points: dict[str, tuple[float, float]] = {}
    for item in filter(None, (s.strip() for s in spec.split(";"))):
        name, sep, xy = item.partition("=")
        name = name.strip()
        try:
            x, y = (float(v) for v in xy.split(","))
        except ValueError:
            raise CalibrationError(f"bad point {item!r}; expected name=x,y") from None
        if not sep or name not in pitch_model.LANDMARKS:
            raise CalibrationError(f"unknown landmark {name!r}; choose from {sorted(pitch_model.LANDMARKS)}")
        if name in points:
            raise CalibrationError(f"landmark {name!r} given twice")
        points[name] = (x, y)
    return points


def session_from_points(points: dict[str, tuple[float, float]]) -> PickerSession:
    fit = [n for n in pitch_model.DEFAULT_FIT_LANDMARKS if n in points]
    missing = [n for n in pitch_model.DEFAULT_FIT_LANDMARKS if n not in points]
    if missing:
        raise CalibrationError(f"missing fit landmarks: {missing}")
    check = [n for n in points if n not in fit]
    return PickerSession(fit_names=fit, check_names=check, points=dict(points))


def build_view(name: str, frame_id: int, session: PickerSession) -> ReferenceView:
    homography = session.fit()

    def observations(names: list[str]) -> list[LandmarkObservation]:
        return [LandmarkObservation(name=n, image=session.points[n], pitch=pitch_model.LANDMARKS[n]) for n in names]

    fit_obs = observations(list(session.fit_names))
    check_obs = observations(list(session.check_points()))
    fit_px, _ = reprojection_errors(
        homography, np.array([o.image for o in fit_obs]), np.array([o.pitch for o in fit_obs])
    )
    check_rms = check_max = None
    if check_obs:
        _, check_m = reprojection_errors(
            homography, np.array([o.image for o in check_obs]), np.array([o.pitch for o in check_obs])
        )
        check_rms, check_max = float(np.sqrt(np.mean(check_m**2))), float(check_m.max())
    return ReferenceView(
        name=name,
        frame_id=frame_id,
        image_to_pitch=homography.H.tolist(),
        fit_points=fit_obs,
        check_points=check_obs,
        fit_rms_px=float(np.sqrt(np.mean(fit_px**2))),
        check_rms_m=check_rms,
        check_max_m=check_max,
    )


def run_calibrate(
    cfg: Config,
    video: Path,
    *,
    frame_id: int = 0,
    view_name: str = "main",
    points: dict[str, tuple[float, float]] | None = None,
    force: bool = False,
    propagate: bool = True,
    propagate_only: bool = False,
    import_views: Path | None = None,
    auto: bool = False,
    auto_step_s: float = 2.0,
) -> CalibrateResult:
    video_id = video_id_from_path(video)
    resize = cfg.video.resize
    with VideoReader(video, resize=resize) as reader:
        meta = reader.meta(video_id)
    cal_path = cfg.calibration_path(video_id)
    out_dir = cfg.output_dir_for(video_id)
    existing = read_yaml_doc(cal_path, CalibrationDoc) if cal_path.is_file() else None

    view = check_path = None
    if propagate_only:
        if existing is None:
            raise CalibrationError(f"{cal_path} does not exist; calibrate a reference view first")
        calibration = existing
    elif auto:
        calibration, check_path = _auto_views(cfg, video, meta, auto_step_s, out_dir)
        write_yaml_doc(calibration, cal_path)
        log.info("wrote %s with %d automatic views", cal_path, len(calibration.views))
    elif import_views is not None:
        calibration = _import_views(cfg, import_views, existing, video_id, (meta.width, meta.height), force)
        write_yaml_doc(calibration, cal_path)
        log.info("wrote %s with %d views", cal_path, len(calibration.views))
    else:
        if existing is not None and not force and any(v.name == view_name for v in existing.views):
            raise CalibrationError(f"view {view_name!r} already exists in {cal_path}; pass --force to replace it")
        if points is None:
            picked = run_picker(
                lambda i: read_frames(video, [i], resize).get(i) if 0 <= i < meta.n_frames else None,
                start_frame=frame_id,
                n_frames=meta.n_frames,
            )
            if picked is None:
                raise CalibrationError("calibration cancelled")
            frame_id, session = picked.frame_id, picked.session
        else:
            session = session_from_points(points)
        view = build_view(view_name, frame_id, session).model_copy(
            update={"source_video": _repo_relative(cfg, video)}
        )

        other_views = [v for v in existing.views if v.name != view_name] if existing else []
        calibration = CalibrationDoc(
            video_id=video_id, image_size=(meta.width, meta.height), views=[*other_views, view]
        )
        write_yaml_doc(calibration, cal_path)
        log.info("wrote %s", cal_path)

        image = read_frames(video, [frame_id], resize)[frame_id]
        check_path = out_dir / "calibration_check.png"
        check_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(check_path), draw_session(image, session, session.fit()))
        log.info("wrote %s", check_path)

    camera = camera_path = None
    if propagate or propagate_only:
        camera = track_video(video, meta, calibration, cfg.camera, resize, root=cfg.root)
        camera_path = write_doc(camera, out_dir / "camera.json")
        log.info("wrote %s", camera_path)
    return CalibrateResult(cal_path, check_path, camera_path, view, camera)


def _auto_views(cfg: Config, video: Path, meta, step_s: float, out_dir: Path) -> tuple[CalibrationDoc, Path]:
    """Reference views without clicks: see cricsense.calibration.auto."""
    step = max(1, int(round(step_s * meta.fps)))
    frames = read_frames(video, range(0, meta.n_frames, step), cfg.video.resize)
    views = find_views(frames)
    if not views:
        raise CalibrationError("no frame shows the pitch's creases clearly enough to calibrate automatically")
    refs = []
    for k, (frame_id, cal, n) in enumerate(views):
        fit = [LandmarkObservation(name=name, image=cal.corners[name], pitch=pitch_model.LANDMARKS[name]) for name in CORNER_NAMES]
        refs.append(
            ReferenceView(
                name=f"auto_{k}",
                source_video=_repo_relative(cfg, video),
                frame_id=frame_id,
                image_to_pitch=cal.image_to_pitch.tolist(),
                fit_points=fit,
                fit_rms_px=0.0,
            )
        )
        log.info("view auto_%d: frame %d, crease score %.3f, %d frames agree", k, frame_id, cal.score, n)
    best = views[0]
    image = frames[best[0]]
    check = out_dir / "calibration_check.png"
    check.parent.mkdir(parents=True, exist_ok=True)
    vis = image.copy()
    P = np.linalg.inv(best[1].image_to_pitch)
    for (x0, y0), (x1, y1) in pitch_model.crease_segments():
        a, b = P @ [x0, y0, 1.0], P @ [x1, y1, 1.0]
        cv2.line(vis, (int(a[0] / a[2]), int(a[1] / a[2])), (int(b[0] / b[2]), int(b[1] / b[2])), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(check), vis)
    return CalibrationDoc(video_id=video_id_from_path(video), image_size=(meta.width, meta.height), views=refs), check


def _repo_relative(cfg: Config, video: Path) -> str:
    path = video.resolve()
    return str(path.relative_to(cfg.root)) if path.is_relative_to(cfg.root) else str(path)


def _find_video(cfg: Config, video_id: str) -> Path:
    matches = sorted(p for p in cfg.paths.video_dir.glob(f"{video_id}.*") if p.is_file())
    if not matches:
        raise CalibrationError(f"no video for {video_id!r} in {cfg.paths.video_dir}")
    return matches[0]


def _import_views(
    cfg: Config,
    source: Path,
    existing: CalibrationDoc | None,
    video_id: str,
    image_size: tuple[int, int],
    force: bool,
) -> CalibrationDoc:
    """Reuse another clip's reference views: same match, same fixed broadcast cameras."""
    donor = read_yaml_doc(source, CalibrationDoc)
    if tuple(donor.image_size) != tuple(image_size):
        raise CalibrationError(f"{source} was calibrated at {donor.image_size}, this video is {image_size}")
    imported = [
        v if v.source_video else v.model_copy(update={"source_video": _repo_relative(cfg, _find_video(cfg, donor.video_id))})
        for v in donor.views
    ]
    kept = list(existing.views) if existing else []
    clash = {v.name for v in kept} & {v.name for v in imported}
    if clash and not force:
        raise CalibrationError(f"views {sorted(clash)} already exist; pass --force to replace them")
    kept = [v for v in kept if v.name not in clash]
    return CalibrationDoc(video_id=video_id, image_size=image_size, views=[*kept, *imported])


def summarize(result: CalibrateResult) -> list[str]:
    lines: list[str] = []
    if result.view is not None:
        v = result.view
        lines.append(f"view {v.name!r} on frame {v.frame_id}: fit rms {v.fit_rms_px:.2f}px")
        if v.check_rms_m is not None:
            lines.append(
                f"held-out check points ({len(v.check_points)}): rms {v.check_rms_m * 100:.1f} cm, "
                f"max {v.check_max_m * 100:.1f} cm"
            )
        else:
            lines.append("no check points given: fit quality is unverified (4 points always fit exactly)")
    if result.camera is not None:
        frames = result.camera.frames
        views = Counter(f.view for f in frames)
        calibrated = len(frames) - views[None]
        run = result.camera.run
        lines.append(
            f"camera: {calibrated}/{len(frames)} frames calibrated "
            f"({100 * calibrated / max(len(frames), 1):.0f}%), by view {dict((k, n) for k, n in views.items() if k)}"
        )
        if run is not None and run.wall_time_s > 0:
            lines.append(f"camera tracking: {run.frames_processed / run.wall_time_s:.1f} frames/s on {run.device}")
    for label, path in (
        ("calibration", result.calibration_path),
        ("check image", result.check_image_path),
        ("camera", result.camera_path),
    ):
        if path is not None:
            lines.append(f"{label}: {path}")
    return lines
