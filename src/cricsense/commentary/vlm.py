"""Shot type and direction from frames around bat contact, by a vision-language model.

Asked for a shot name directly, the models tried here guess (one called every shot a
"sweep", including upright drives it described correctly when asked what it saw). So the
model only describes what is visible - knee on the ground, stance, front or back foot, how
the bat finishes, which side the ball goes - and the shot is named from that by the rules
below, which are the textbook definitions. What comes out is marked shot_source "vlm" in the
facts, never presented as a measurement; if the posture is unclear nothing is named.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from cricsense.commentary.llm import Cache, LLMConfig, complete, parse_json
from cricsense.io import VideoReader

FRAME_OFFSETS = (-2, 2, 6, 12)  # around bat contact
WIDTH = 960
SIDES = ("off", "leg", "straight")

PROMPT = """These are frames from a cricket broadcast, in time order, around the moment the batter plays the ball.
The first frames are close crops on the batter; the last is the wide view just after the shot.
Describe only what you can see of the batter - do not name the shot.
Return JSON only:
{"knee_on_ground": true | false | null, "stance": "upright" | "crouched" | "kneeling" | "unclear",
 "foot": "front" | "back" | "unclear", "bat_finish": "high" | "low" | "unclear",
 "ball_side": "off" | "leg" | "straight" | "unclear", "clear": 0.0-1.0 (how clearly the frames show all this)}"""

# For a provider that takes fewer images per request than we send, the frames go as one picture.
TILED = "\nThe frames are tiled into one picture, numbered in time order: left to right, then top to bottom."

MIN_CLEAR = 0.5


def name_shot(o: dict) -> str | None:
    """The shot, by the textbook, from what the model saw; None if it cannot be told."""
    side, foot = o.get("ball_side"), o.get("foot")
    high = o.get("bat_finish") == "high"
    kneeling = o.get("knee_on_ground") is True or o.get("stance") == "kneeling"
    if side not in SIDES:
        return None
    if kneeling:
        return {"leg": "slog sweep" if high else "sweep", "off": "reverse sweep", "straight": "slog sweep"}[side]
    if o.get("stance") not in ("upright", "crouched"):
        return None
    if foot == "front":
        return {"straight": "straight drive", "off": "cover drive", "leg": "flick"}[side]
    if foot == "back":
        return {"straight": "back-foot punch", "off": "cut" if not high else "back-foot punch", "leg": "pull"}[side]
    return {"straight": "straight drive", "off": "drive", "leg": "clip to leg"}[side]


CROP = 640  # side of the square crop around the batter, in the frame's pixels


def tile(frames: list[bytes], cell: int = 768) -> bytes:
    """The frames as one JPEG, two to a row in time order, each numbered and fitted into its cell."""
    imgs = [cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) for f in frames]
    cols = 2
    grid = np.zeros((-(-len(imgs) // cols) * cell, cols * cell, 3), np.uint8)
    for i, im in enumerate(imgs):
        scale = cell / max(im.shape[:2])
        im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        row, col = divmod(i, cols)
        y0, x0 = row * cell + (cell - im.shape[0]) // 2, col * cell + (cell - im.shape[1]) // 2
        grid[y0 : y0 + im.shape[0], x0 : x0 + im.shape[1]] = im
        cv2.putText(grid, str(i + 1), (col * cell + 12, row * cell + 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
    return cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def contact_frames(video, centre: int, striker_box=None) -> list[bytes]:
    """JPEGs around bat contact. striker_box(frame) -> (x1, y1, x2, y2) | None: where a box is
    known the frame is cropped on the batter (at full resolution the batter is ~250 px tall;
    a whole frame shrunk for the model leaves ~100 px); the last frame stays wide, for direction."""
    wanted = {centre + d for d in FRAME_OFFSETS}
    last = max(wanted)
    out = {}
    with VideoReader(video, start=max(0, min(wanted)), end=last + 1) as reader:
        for f in reader:
            if f.frame_id not in wanted:
                continue
            img = f.image[: int(0.85 * f.image.shape[0])]  # leave out the score bar
            box = striker_box(f.frame_id) if striker_box and f.frame_id != last else None
            if box is not None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                half = max(CROP, int(2.2 * (box[3] - box[1]))) // 2
                h, w = img.shape[:2]
                x0, y0 = int(np.clip(cx - half, 0, max(w - 2 * half, 0))), int(np.clip(cy - half, 0, max(h - 2 * half, 0)))
                img = img[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
                img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA if img.shape[0] > 768 else cv2.INTER_CUBIC)
            else:
                h, w = img.shape[:2]
                img = cv2.resize(img, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
            out[f.frame_id] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
    return [out[k] for k in sorted(out)]


VOTES = 3  # readings per shot
AGREE = 2  # readings that must name the same shot


def read_shot(cfg: LLMConfig, key: str, frames: list[bytes], cache: Cache, votes: int = VOTES) -> dict | None:
    """{"shot", "direction", "observed", "votes"} if enough independent readings agree, else None.

    One reading is not stable: with the cache cleared, the same frames gave a different shot
    (Axar's six: "slog sweep", then "cover drive"). So the posture is read `votes` times and a
    shot is reported only when AGREE of them name it (and the side) the same way."""
    if not frames:
        return None
    digest = [np.frombuffer(f, np.uint8).sum() % 1_000_003 for f in frames]
    prompt = PROMPT
    if cfg.max_images is not None and len(frames) > cfg.max_images:
        frames, prompt = [tile(frames)], PROMPT + TILED
    readings = []
    for attempt in range(votes):
        k = cache.key("vlm-posture", cfg.provider, cfg.model, prompt, digest, attempt)
        reply = cache.get(k)
        if reply is None:
            reply = complete(cfg, key, "You describe cricket video frames. Answer only with the JSON asked for.", prompt, frames)
            cache.put(k, reply)
        try:
            o = parse_json(reply)
            clear = float(o.get("clear", 0))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        shot = name_shot(o)
        if shot is not None and clear >= MIN_CLEAR:
            readings.append((shot, o["ball_side"], o))
    if not readings:
        return None
    counts: dict[tuple[str, str], int] = {}
    for shot, side, _ in readings:
        counts[(shot, side)] = counts.get((shot, side), 0) + 1
    (shot, side), n = max(counts.items(), key=lambda kv: kv[1])
    if n < AGREE:
        return None
    observed = next(o for s_, sd, o in readings if (s_, sd) == (shot, side))
    return {"shot": shot, "direction": f"{side} side" if side != "straight" else "straight", "observed": observed, "votes": f"{n}/{votes}"}
