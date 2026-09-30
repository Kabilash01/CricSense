"""`cricsense render`: overlay.mp4 and pitch_map.png from the pass outputs.

CPU only. Frames are piped to ffmpeg (H.264, the source audio copied) so the result plays in
any browser; without ffmpeg it falls back to OpenCV's mp4v writer, video only.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from cricsense.config import Config
from cricsense.io import VideoReader, read_doc
from cricsense.render.overlay import Scene, compose
from cricsense.render.pitch_map import summary_png
from cricsense.render.wagon_wheel import Shot, draw_wagon_wheel
from cricsense.commentary.llm import WORDS_PER_S
from cricsense.schema import BallDoc, CameraDoc, CommentaryDoc, EventsDoc, EventType, PlayersDoc, PoseDoc

log = logging.getLogger("cricsense.render")


def has_audio(path: Path) -> bool:
    if not shutil.which("ffprobe"):
        return False
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout
    return bool(out.strip())


def audio_source(cfg: Config, video_id: str, video: Path) -> tuple[Path, float] | None:
    """The clip's own sound, else its source broadcast's at the clip's offset (configs/clips.yaml)."""
    if has_audio(video):
        return video, 0.0
    clips = cfg.root / "configs" / "clips.yaml"
    if clips.is_file():
        entry = ((yaml.safe_load(clips.read_text()) or {}).get("clips") or {}).get(video_id)
        if entry:
            src = cfg.root / entry["source"]
            if src.is_file() and has_audio(src):
                return src, float(entry["start_s"])
    return None


class _Writer:
    def __init__(
        self,
        path: Path,
        fps: float,
        size: tuple[int, int],
        audio: tuple[Path, float] | None,
        start_s: float,
        voice: Path | None = None,
    ) -> None:
        self.proc = None
        self.cv = None
        w, h = size
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
            if audio is not None:
                cmd += ["-ss", f"{audio[1] + start_s:.3f}", "-i", str(audio[0])]
            if voice is not None:
                cmd += ["-ss", f"{start_s:.3f}", "-i", str(voice)]
            if audio is not None and voice is not None:
                # the crowd ducks under the commentator (sidechain), then both are mixed
                cmd += [
                    "-filter_complex",
                    "[2:a]aresample=48000,asplit=2[sc][cm];[1:a:0]aresample=48000[bg];"
                    "[bg][sc]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=450[duck];"
                    "[duck][cm]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95[a]",
                    "-map", "0:v", "-map", "[a]",
                ]
            elif audio is not None or voice is not None:
                cmd += ["-map", "0:v", "-map", "1:a:0"]
            if audio is not None or voice is not None:
                cmd += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            log.warning("ffmpeg not found: writing mp4v without audio")
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame) -> None:
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg failed while writing the overlay")
        else:
            self.cv.release()


def pitch_map_rows(scene: Scene) -> list[tuple[str, ...]]:
    rows = []
    for dv in scene.deliveries:
        ball, _, who = dv.title.partition("  ") if "\u2192" in dv.title else (str(dv.did + 1), "", dv.title)
        if dv.length_m is None:
            length, line = "not seen", ""
        else:
            length = f"{dv.length_class} {dv.length_m:.1f} m" + ("" if dv.confirmed else "*")
            line = f"{abs(dv.line_m) * 100:.0f} cm {'R' if dv.line_m > 0 else 'L'}"
        speed = f"{dv.speed} {dv.speed_source}" if dv.speed else ""
        rows.append((ball, who.title().replace("\u2192", "\u2192"), length, line, speed, dv.outcome or ""))
    return rows


def run_render(
    cfg: Config,
    video_id: str,
    start_s: float | None = None,
    end_s: float | None = None,
    out_name: str = "overlay.mp4",
    voice: bool = True,
) -> tuple[Path, Path, int]:
    out_dir = cfg.output_dir_for(video_id)
    ball = read_doc(out_dir / "ball.json", BallDoc)
    events = read_doc(out_dir / "events.json", EventsDoc)
    players = read_doc(out_dir / "players.json", PlayersDoc) if (out_dir / "players.json").is_file() else None
    camera = read_doc(out_dir / "camera.json", CameraDoc) if (out_dir / "camera.json").is_file() else None
    video = next(cfg.paths.video_dir.glob(f"{video_id}.*"))
    fps = ball.video.fps
    size = (ball.video.width, ball.video.height)
    scene = Scene(ball, events, players, camera, size)
    hands: dict[int, str] = {}
    if (out_dir / "pose.json").is_file():
        hands = {b.delivery_id: b.batting_hand for b in read_doc(out_dir / "pose.json", PoseDoc).batters if b.batting_hand}
    if (out_dir / "commentary.json").is_file():
        facts = read_doc(out_dir / "commentary.json", CommentaryDoc).facts
        # one batting hand per batter: the majority over their deliveries
        by_name: dict[str, list[str]] = {}
        for fact in facts:
            if fact["delivery_id"] in hands and fact.get("striker"):
                by_name.setdefault(fact["striker"], []).append(hands[fact["delivery_id"]])
        hand_of = {n: max(set(v), key=v.count) for n, v in by_name.items()}
        for fact in facts:
            if fact.get("shot"):
                scene.shots[fact["delivery_id"]] = f"{fact['shot']}, {fact['direction']}"
            hand = hand_of.get(fact.get("striker"))
            if fact.get("shot") and hand and (fact.get("off_bat") or 0) > 0:
                scene.wagon.append((fact["delivery_id"], Shot(fact["over"] or str(fact["delivery_id"] + 1), fact["shot"], fact["off_bat"], hand)))
        if scene.wagon:
            ww = draw_wagon_wheel([s for _, s in scene.wagon], 720, bg_alpha=255)
            canvas = np.full((ww.shape[0], ww.shape[1], 3), (22, 24, 26), np.uint8)
            a = ww[..., 3:4] / 255.0
            canvas[:] = (ww[..., :3] * a + canvas * (1 - a)).astype(np.uint8)
            cv2.imwrite(str(out_dir / "wagon_wheel.png"), canvas)
    if (out_dir / "pose.json").is_file():
        pose = read_doc(out_dir / "pose.json", PoseDoc)
        scene.skeletons = {p.frame_id: np.array(p.keypoints) for p in pose.frames}
        for b in pose.batters:
            bits = []
            if b.footwork:
                bits.append(b.footwork + (f" {abs(b.stride_m):.1f} m" if b.footwork != "in the crease" and b.stride_m else ""))
            if b.swing:
                bits.append({"played": "played at it", "left": "left it"}[b.swing])
            if bits:
                scene.batter[b.delivery_id] = ", ".join(bits)
    voice_wav = out_dir / "commentary.wav"
    use_voice = voice and voice_wav.is_file() and (out_dir / "voice.json").is_file()
    if use_voice:
        scene.captions = [(c["start_s"], c["end_s"], c["text"]) for c in json.loads((out_dir / "voice.json").read_text())["lines"]]
    elif (out_dir / "commentary.json").is_file():  # not spoken yet: captions at the planned times
        lines = read_doc(out_dir / "commentary.json", CommentaryDoc).lines
        scene.captions = [(ln.t, ln.t + min(ln.max_s, len(ln.text.split()) / WORDS_PER_S + 1.5), ln.text) for ln in lines]

    title = f"Pitch map - {video_id}"
    if any(e.type == EventType.RELEASE and "bowler" in e.payload for e in events.events):
        bowlers = sorted({e.payload["bowler"].title() for e in events.events if e.type == EventType.RELEASE and "bowler" in e.payload})
        title += "  (" + ", ".join(bowlers) + ")"
    map_path = out_dir / "pitch_map.png"
    cv2.imwrite(str(map_path), summary_png(scene.map_points, pitch_map_rows(scene), title))

    start = int(round((start_s or 0.0) * fps))
    end = ball.video.n_frames if end_s is None else min(ball.video.n_frames, int(round(end_s * fps)))
    out_path = out_dir / out_name
    t0 = time.perf_counter()
    writer = _Writer(out_path, fps, size, audio_source(cfg, video_id, video), start / fps, voice_wav if use_voice else None)
    n = 0
    try:
        with VideoReader(video, start=start, end=end) as reader:
            for fr in reader:
                writer.write(compose(fr.image, fr.frame_id, scene))
                n += 1
    finally:
        writer.close()
    log.info("wrote %s: %d frames in %.0f s (%.1f fps)", out_path, n, time.perf_counter() - t0, n / max(time.perf_counter() - t0, 1e-6))
    return out_path, map_path, n
