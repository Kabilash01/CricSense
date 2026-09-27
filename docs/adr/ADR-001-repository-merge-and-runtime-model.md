# ADR-001: Repository Merge and Runtime Execution Model

**Status:** Proposed
**Date:** 2026-09-26
**Deciders:** Kabi (author), internship supervisor / lab examiner

---

## Context

CricSense currently exists as two independent repositories:

| Repo | Scope | Real code | State |
|------|-------|-----------|-------|
| `CricSense-AI` | Player detection, tracking, stable IDs, jersey identity, event timeline | ~1,800 LOC in `src/` | YOLOv8m detector trained to mAP50 **0.912** / mAP50-95 **0.598**; identity layer broken |
| `CricSense-ball-tracking-and-Tracjectory-` | Ball detection, Kalman tracking, bounce, pitch length, boundary | ~1,520 LOC | Four near-duplicate entry scripts; speed calibration physically invalid |

Forces at play:

1. **Shared calibration.** Both halves need a pixel → pitch homography derived from the stumps. Ball line/length is meaningless without it, and player *role* (striker / non-striker / bowler / keeper) can only be derived from crease geometry. Today `events/homography.py` is a 26-byte comment stub and neither repo has it.
2. **Cross-domain events.** Ball–bat contact requires the ball trajectory *and* the batter's bounding box. Neither repo can compute it alone.
3. **Duplication already exists.** The Kalman filter is implemented three times (`ball/src/main.py`, `ball/src/ball_tracking.py`, `ball/src/tracking/kalman_filter.py`) and a fourth time in `CricSense-AI/src/vision/tracking/ball_kalman.py`. Geometry helpers, JSON exporters, and config loading are duplicated across repos.
4. **Hard deadline.** The system must be demonstrable at a lab exam, on a single laptop RTX GPU, from a clean clone.
5. **Two 0.9-scale models.** Running YOLOv8m twice per frame is roughly additive on the GPU (~20–25 ms each at 640px FP32), plus a SmolVLM `generate()` call currently sitting *inside* the frame loop in `real_optimized.py`.

The repository question ("merge or not") and the runtime question ("one loop or two") are independent and are decided separately below.

---

## Decision

**Merge both repositories into a single Python package `cricsense`, and execute as sequential passes over the video rather than one combined realtime loop.**

Players and ball each run their own pass and write a JSON artifact conforming to a shared schema. A third, CPU-only fusion pass joins them into events. A combined realtime loop is retained as an optional, explicitly degraded `live` mode.

---

## Options Considered

### Option A: Keep two repositories, fix each independently

| Dimension | Assessment |
|-----------|------------|
| Complexity | Low per repo, High overall |
| Cost | Homography, config, schema, CI implemented twice |
| Scalability | Poor — cross-domain events have no home |
| Team familiarity | Highest (no restructuring) |

**Pros:** No migration work. Each repo keeps its git history. Lowest immediate risk.
**Cons:** Ball–bat contact and no-ball detection are homeless — they belong to neither repo. Calibration must be duplicated and kept in sync by hand. The examiner sees two half-finished projects instead of one system. Duplicate Kalman implementations stay duplicated and continue to drift (they already use **opposite sign conventions** for bounce detection).

---

### Option B: Merge into one package, sequential passes *(chosen)*

| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium (one-time migration) |
| Cost | ~1 day of restructuring |
| Scalability | Good — new stages are new passes |
| Team familiarity | Medium |

**Pros:** Calibration written once and consumed by both halves. Each pass runs at full GPU speed with no contention. Every stage is independently debuggable and independently benchmarkable, which produces the per-stage metrics a viva rewards. Fusion is pure CPU and runs in milliseconds. A failure in the ball pass does not take down the player pass. Matches the intent of the original block diagram, where the score engine consumes structured event JSON rather than raw frames.
**Cons:** Not "realtime" in the literal sense — requires 2 passes over the video. Intermediate JSON files must be versioned and schema-checked. Slightly more disk I/O.

---

### Option C: Merge into one package, single combined realtime loop

| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Cost | Significant optimisation work (TensorRT, threading, ROI cropping) |
| Scalability | Poor — every new model taxes the same frame budget |
| Team familiarity | Low |

**Pros:** Genuinely realtime; single command; demos well on live footage.
**Cons:** Two YOLOv8m forward passes land near 40–50 ms/frame before tracking and rendering — roughly 20–25 FPS, and that is *before* the VLM. Debugging is far harder because a ball failure and an identity failure appear in the same log. Requires FP16/TensorRT export, ROI-cropped ball inference, player-model striding, and moving the VLM to a worker thread just to reach an acceptable frame rate. High risk against a fixed exam deadline.

---

### Option D (rejected outright): Retrain one model for players + ball + stumps

Rejected without full analysis. The ball is a small, motion-blurred object that needs high input resolution; players are large objects well served at 640px. A single model at a single input size serves both poorly, and this would discard an existing detector at mAP50 0.912 that took 65 epochs (~7.5 hours) to train.

---

## Trade-off Analysis

The decisive trade-off is **latency versus debuggability under a deadline.**

Option C buys literal realtime performance at the cost of entangling two failure domains that are currently *both* broken. The identity layer produces 534 stable IDs for a clip containing roughly 13 people, and the ball pipeline reports a release speed of 409 km/h. Debugging two broken subsystems inside one frame loop, under exam pressure, is the worst available position.

Option B decouples them for free. Nothing about the current requirement set actually demands realtime: the deliverable is an analysed clip with an overlay and an event log, not a live broadcast feed. "Realtime-capable" can be demonstrated by reporting per-stage FPS measured in isolation, which is a more honest number than a combined figure inflated by frame-skipping.

Option A is cheapest today and most expensive by the end of the week, because the first genuinely cross-domain feature (ball–bat contact) forces the merge anyway — at that point mid-refactor rather than at a clean starting line.

Option C is therefore retained as an *optional mode* rather than discarded, so the capability can be demonstrated without the core deliverable depending on it.

---

## Consequences

**What becomes easier**
- Homography is implemented once (`cricsense/calibration/`) and consumed by role assignment, line/length, and no-ball logic alike.
- Cross-domain events (ball–bat contact, no-ball, run-out) have a natural home in the fusion pass.
- Each stage gets its own metric: detection mAP, tracking IDF1/ID-switches, identity accuracy, ball recall — presentable as a table.
- One `requirements.txt`, one config system, one `README`. Currently both `requirements.txt` files and all three `ball/configs/*.yaml` files are empty.
- Duplicate Kalman implementations collapse to one, resolving the conflicting bounce sign conventions.

**What becomes harder**
- Intermediate JSON becomes a contract. A schema change breaks downstream passes, so `schema/` must be versioned and validated (Pydantic).
- Two passes mean roughly 2× wall-clock over a full clip. Acceptable for clips of 30–120 s; would not be for a live feed.
- Git history from the two source repos is not preserved by a plain file copy (see SETUP.md for the subtree-import alternative).

**What we'll need to revisit**
- If the project later targets live RTSP ingest, Option C returns and the optimisation work deferred here becomes mandatory.
- The 1.1 GB `jersey_crops/` directory must not be carried into the merged repo; it is regenerable output and belongs in `.gitignore`.
- If identity accuracy stays weak after the geometry fix, an appearance-embedding re-ID model (OSNet) becomes a separate ADR.

---

## Action Items

1. [ ] Create the `cricsense` package skeleton and shared Pydantic schema (`schema/models.py`).
2. [ ] Migrate the player half per the mapping table in `SETUP.md`; drop `bytetrack_wrapper.py` (imports `yolox`, unused — the real path uses Ultralytics' built-in tracker).
3. [ ] Migrate the ball half; collapse `main.py`, `ball_tracking.py`, `test.py`, `test2.py` into one `ball` pass with a single Kalman and a single bounce sign convention.
4. [ ] Implement `calibration/homography.py` for real — 4 stump/crease correspondences → pitch coordinates, persisted per video.
5. [ ] Fix `PlayerRegistry`: replace role-equality matching with IoU + appearance gating, and fix `self.jersey_logged = set` → `set()`.
6. [ ] Derive role from pitch geometry, not from the YOLO class name.
7. [ ] Wire `scoreboard_ocr.py` into the identity pass as the primary name source; demote jersey OCR to a secondary vote.
8. [ ] Remove the km/h claim from the ball pass until homography-based speed exists; report line and length on a pitch map instead.
9. [ ] Add `.gitignore` entries for `models/`, `data/`, `outputs/`, `jersey_crops/`.
10. [ ] Write per-stage benchmark script and record the metrics table for the viva.
