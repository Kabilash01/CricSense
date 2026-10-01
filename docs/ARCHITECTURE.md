# CricSense — Architecture (original design)

> **This is the design written before the build (September 2026).** The system as built -
> automatic calibration, the score engine, commentary, voice, storage and API, live mode -
> and what was measured is described in [REPORT.md](REPORT.md); what was tried and dropped,
> with its measurements, in [decisions.md](decisions.md). Kept for the record of the plan.

**Audience:** anyone picking up this codebase, including future-you before the lab exam.
**Companion docs:** [ADR-001](adr/ADR-001-repository-merge-and-runtime-model.md) for *why* this shape, [SETUP-merge.md](history/SETUP-merge.md) for *how* it was built from the two source repos.

---

## 1. Context and goals

CricSense analyses cricket video and produces a structured event log plus an annotated overlay.

**In scope**
- Detect and track players, stumps, and the ball.
- Assign each tracked player a stable identity and a cricket **role** (striker, non-striker, bowler, keeper, fielder, umpire).
- Reconstruct the ball's trajectory, detect the bounce, and report line and length in pitch coordinates.
- Emit a timestamped event log and render an overlay video.

**Explicitly out of scope (for the exam build)**
- Live RTSP ingest and true realtime operation.
- Authoritative scoring, DRS-grade ball tracking, or broadcast commentary generation.
- Absolute ball speed in km/h. A single uncalibrated camera cannot measure it; see §6.

**Constraints**
- One laptop RTX GPU (8 GB class).
- Must run from a clean clone with no hardcoded absolute paths.
- Deadline-bound: correctness of a few stages beats breadth of many.

---

## 2. High-level design

Three passes over one video, joined by files. Each pass is independently runnable and independently benchmarkable.

```
                         configs/calibration/<video_id>.yaml
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
┌───────▼────────┐           ┌────────▼───────┐           ┌─────────▼────────┐
│  PASS 1        │           │  PASS 2        │           │  PASS 3          │
│  players       │           │  ball          │           │  fuse            │
│  (GPU)         │           │  (GPU)         │           │  (CPU, ms)       │
├────────────────┤           ├────────────────┤           ├──────────────────┤
│ detect         │           │ detect (ROI)   │           │ join by frame_id │
│ track          │           │ kalman         │           │ ball-bat contact │
│ stable ID      │           │ trajectory fit │           │ bounce → length  │
│ role (geometry)│           │ bounce         │           │ boundary / wicket│
│ identity (OCR) │           │ line & length  │           │ event timeline   │
└───────┬────────┘           └────────┬───────┘           └─────────┬────────┘
        │                             │                             │
        ▼                             ▼                             ▼
 outputs/<vid>/players.json   outputs/<vid>/ball.json     outputs/<vid>/events.json
                                                                    │
                                                          ┌─────────▼────────┐
                                                          │  PASS 4  render  │
                                                          │  overlay.mp4     │
                                                          │  pitch_map.png   │
                                                          └──────────────────┘
```

A `live` mode runs passes 1–3 in a single loop for demonstration. It is a degraded path, not the primary deliverable — see §7.

---

## 3. Package layout

```
cricsense/
├── README.md
├── SETUP.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml              # paths, thresholds, device
│   ├── players.yaml              # detector + tracker params
│   ├── ball.yaml                 # detector + kalman params
│   └── calibration/              # one YAML per video (homography)
├── models/                       # .pt weights — GITIGNORED
├── data/                         # source clips — GITIGNORED
├── outputs/                      # all generated artifacts — GITIGNORED
├── docs/
│   ├── ARCHITECTURE.md
│   └── adr/ADR-001-...md
├── src/cricsense/
│   ├── cli.py                    # single entry point, subcommands
│   ├── config.py                 # YAML → dataclass, no hardcoded paths
│   ├── schema/models.py          # Pydantic contracts (§4)
│   ├── io/
│   │   ├── video_reader.py
│   │   └── writers.py
│   ├── calibration/
│   │   ├── homography.py         # pixel ↔ pitch, 4-point correspondence
│   │   └── pitch_model.py        # crease/stump constants in metres
│   ├── detect/
│   │   ├── player_detector.py    # YOLOv8m, mAP50 0.912
│   │   └── ball_detector.py      # YOLO + ROI crop around prediction
│   ├── track/
│   │   ├── player_tracker.py     # Ultralytics ByteTrack
│   │   └── ball_kalman.py        # ONE Kalman, ONE sign convention
│   ├── identity/
│   │   ├── registry.py           # stable IDs — IoU + appearance gated
│   │   ├── role_assigner.py      # role from pitch geometry
│   │   ├── scoreboard_ocr.py     # PRIMARY identity source
│   │   ├── jersey_ocr.py         # secondary vote
│   │   └── vote_tracker.py       # temporal voting + locking
│   ├── ball/
│   │   ├── trajectory.py         # two-parabola fit, split at bounce
│   │   ├── bounce.py
│   │   └── line_length.py        # in pitch coords, not image coords
│   ├── fuse/
│   │   ├── contact.py            # ball ∩ batter box + trajectory kink
│   │   └── events.py             # event timeline assembly
│   └── render/
│       ├── overlay.py
│       └── pitch_map.py          # 2D top-down line/length plot
├── scripts/benchmark.py          # per-stage metrics for the viva
└── tests/
```

**Rule:** nothing in `src/cricsense/` opens a file by absolute path. All paths come from `config.py`.

---

## 4. Data contracts

Every pass validates its input and output against `schema/models.py`. This is what makes the passes independent.

```python
class PlayerFrame(BaseModel):
    frame_id: int
    t: float                      # seconds
    sid: int                      # stable id
    box: tuple[int, int, int, int]
    role: Role                    # enum, from geometry
    pitch_xy: tuple[float, float] | None   # metres, None if uncalibrated
    identity: str | None          # from scoreboard, else None
    identity_conf: float

class BallFrame(BaseModel):
    frame_id: int
    t: float
    uv: tuple[float, float]       # image coords
    observed: bool                # False = Kalman prediction only
    pitch_xy: tuple[float, float] | None

class Event(BaseModel):
    frame_id: int
    t: float
    type: EventType               # RELEASE | BOUNCE | CONTACT | BOUNDARY | WICKET
    actors: dict[str, int]        # role → sid
    payload: dict                 # length class, line offset, etc.
    confidence: float
    source: str                   # which module emitted it
```

`source` and `confidence` on every event exist so the overlay can visually distinguish a confident measurement from a heuristic guess — which is exactly the question an examiner asks.

---

## 5. Key decisions and trade-offs

### 5.1 Role comes from geometry, never from the detector class

The current code sets role via `infer_role_from_class(det["name"])` — straight from the YOLO label. A detector cannot reliably distinguish a batsman from a bowler by appearance; the difference is *positional and temporal*. The class flips frame to frame, and because `PlayerRegistry.resolve_id` requires role equality to match a track, every flip mints a new stable ID. That is the direct cause of **534 stable IDs for ~13 people**.

Replacement: project each player's foot point through the homography into pitch coordinates, then assign role by region — striker's crease, non-striker's crease, behind the stumps, run-up corridor. Role becomes an *output* of calibration, not an input to identity.

### 5.2 Identity is fused, not OCR'd

Jersey numbers are structurally unavailable from the main broadcast angle: the camera sits behind the bowler, so the batter faces it (no number) and the bowler's back is visible only briefly and blurred. Therefore:

| Source | Role | Why |
|--------|------|-----|
| Scoreboard OCR | **Primary** | Names striker (`*`) and bowler explicitly, rendered in clean synthetic text |
| Appearance cluster | Team split | Separates batting side, fielding side, umpires |
| Jersey OCR | Secondary vote | Only on crops passing a legibility gate, voted across the tracklet |

`scoreboard_ocr.py` already exists and works — it is simply never called. Wiring it in is the single highest-value fix in the project.

### 5.3 VLM stays out of the frame loop

`real_optimized.py` calls `SmolVLMJerseyReader.read_jersey_number()` inline, inside the per-detection loop. A VLM `generate()` call costs tens to hundreds of milliseconds; this alone dominates the frame budget. Crops are queued and processed after the pass (or in a worker thread in `live` mode).

SmolVLM-256M is also the wrong model for small blurred digits. A dedicated OCR (EasyOCR/PaddleOCR) with a legibility classifier in front of it is both faster and more accurate here. MoE VLMs were considered and rejected: MoE reduces compute per token but not resident memory, so a 30B-A3B model does not fit the target GPU.

### 5.4 One Kalman, one sign convention

Three Kalman implementations exist today, and the bounce test disagrees between them — `main.py` uses `prev_vy < 0 and vy > 0` while `tracker.py` uses `y2 > y1 and y3 < y2`. In image coordinates y increases downward, so these are opposite tests; one is wrong. The merged package has exactly one implementation, documented convention: **image y increases downward; a bounce is a local maximum in y.**

### 5.5 Fusion is CPU and cheap

Ball–bat contact = ball trajectory entering the batter's box AND a discontinuity in the fitted trajectory. Both inputs are already in the JSON. This is milliseconds of work and needs no GPU, which is why it deserves its own pass rather than a slot in the frame loop.

---

## 6. Known limits (state these in the viva before you are asked)

1. **Speed is not measured.** `METERS_PER_PIXEL = 18.5/520` plus a `perspective_scale()` fudge is not calibration. The existing output contains `release_speed_kmph: 409.9` — caused by `release_speed.py` assigning `dist_px / time_sec` to a field labelled km/h with no unit conversion. Until homography-based speed exists, the system reports **line and length**, which are geometrically recoverable, and does not report km/h.
2. **Length classification was image-relative.** Thresholding normalised image `y` (`> 0.75 → YORKER`) encodes the camera, not the pitch. After calibration, length is measured in metres from the stumps.
3. **Single camera = no true 3D.** Trajectory is a plane-constrained estimate. Say so.
4. **Broadcast cuts break tracking.** Shot-boundary detection belongs in ingest; without it, replays are double-counted and homography silently goes stale.

---

## 7. `live` mode (optional)

Runs passes 1–3 in one loop. Expect roughly 20–25 FPS for two YOLOv8m forward passes at 640px FP32 on a laptop RTX, before tracking and drawing. Mitigations, in order of payoff:

1. VLM/OCR moved to a worker thread (largest single win).
2. Player detector strided — every 3rd frame, ByteTrack interpolates. The ball must run every frame.
3. Ball detector on an ROI crop around the Kalman prediction, not the full frame. Faster *and* fewer crowd false positives.
4. FP16 + TensorRT export (`model.export(format="engine", half=True)`), typically 2–3× on RTX.

`live` mode is demonstrated, not depended upon. The graded deliverable is the three-pass output.

---

## 8. Metrics to report

| Stage | Metric | Current |
|-------|--------|---------|
| Player detection | mAP50 / mAP50-95 | **0.912 / 0.598** |
| Player tracking | IDF1, ID switches | not yet measured |
| Identity | stable IDs vs. true persons | **534 vs ~13** — the headline bug |
| Ball detection | recall on ball-visible frames | not yet measured |
| Bounce | frame error vs. hand-labelled | not yet measured |
| Throughput | FPS per pass | not yet measured |

Filling this table is worth more at a lab exam than adding a fourth half-built layer.
