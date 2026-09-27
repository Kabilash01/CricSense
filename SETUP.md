# SETUP — building the merged repo from the two source repos

This is the migration runbook. Follow it top to bottom once; after that use [README.md](README.md).

**Prerequisites:** Python 3.10+, CUDA-capable GPU, both source repos cloned, the trained weights on disk.

---

## Step 0 — Make the working folder

```bash
mkdir cricsense && cd cricsense
git init
git clone --depth 1 https://github.com/Kabilash01/CricSense-AI.git                      _src_ai
git clone --depth 1 https://github.com/Kabilash01/CricSense-ball-tracking-and-Tracjectory- _src_ball
```

`_src_*` are staging folders. They get deleted in Step 6 and must never be committed.

> **History note:** a plain copy loses the two repos' git history. If history matters for the internship report, use `git subtree add --prefix=legacy/ai <url> main` instead and migrate files out of `legacy/`. For the exam build, a clean history is fine and simpler.

---

## Step 1 — Skeleton and ignore rules

Create `.gitignore` **first**, before copying anything. `_src_ai/jersey_crops/` is **1.1 GB** of regenerable crops; it must never enter the new repo.

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/

models/
data/
outputs/
jersey_crops/
*.pt
*.pth
*.engine
*.mp4
*.avi
```

Then:

```bash
mkdir -p src/cricsense/{schema,io,calibration,detect,track,identity,ball,fuse,render}
mkdir -p configs/calibration models data outputs docs/adr scripts tests
find src/cricsense -type d -exec touch {}/__init__.py \;
```

---

## Step 2 — Migrate the player half

| From `_src_ai/` | To | Action |
|---|---|---|
| `src/vision/detectors/yolov8_detector.py` | `detect/player_detector.py` | Keep. Remove the hardcoded `configs/bytetrack.yaml` path resolution; take it from config. |
| `src/ingest/video_reader.py` | `io/video_reader.py` | Keep. Delete `video_reader1.py` (14-line duplicate). |
| `src/core/player_registry.py` | `identity/registry.py` | **Rewrite** — see Step 5. |
| `src/core/event_timeline.py` | `fuse/events.py` | Merge into the event assembler. |
| `src/jersey/jersey_vote_tracker.py` | `identity/vote_tracker.py` | Keep as-is; the voting logic is sound. |
| `src/ocr/scoreboard_ocr.py` | `identity/scoreboard_ocr.py` | Keep and **actually call it** — currently dead code. |
| `src/ocr/jersey_ocr.py` | `identity/jersey_ocr.py` | Keep as the secondary vote. |
| `src/core/ball_release_detector.py` | `fuse/events.py` | Fold in as a RELEASE emitter. |
| `src/utils/{config,logger}.py` | `config.py`, `io/writers.py` | Expand; `config.py` is 16 lines today. |
| `yolov8m_production4/` | keep at repo root | Training curves — evidence for the viva. Keep the PNGs and `results.csv`; the weights go to `models/` (gitignored). |

**Do not migrate** (dead or stub):

```
src/vision/tracking/bytetrack_wrapper.py   # imports yolox; unused, real path is Ultralytics
src/jersey/smolvlm_reader.py               # 256M VLM, wrong tool for blurred digits
src/pipeline/{main,test_model,realtime_pipeline,verifier_worker,active_learning}.py
src/events/{homography,wicket_detection,run_detection,heuristics}.py   # ≤40-byte stubs
src/llm/, src/tts/, src/api/, src/ui/      # comment-only stubs
src/vision/{segmentation,fusion}/, src/vision/tracking/optical_flow.py
src/pose/                                  # revisit only if no-ball detection is attempted
jersey_crops/                              # 1.1 GB of regenerable output
```

`src/pipeline/real_optimized.py` is the reference for the new players pass — read it, then rewrite it against the new schema. Do not copy it: the VLM call sits inside the detection loop and must come out.

---

## Step 3 — Migrate the ball half

| From `_src_ball/` | To | Action |
|---|---|---|
| `src/detection/yolo_detector.py` | `detect/ball_detector.py` | Keep the area/ratio filters; add ROI cropping around the Kalman prediction. |
| `src/tracking/kalman_filter.py` | `track/ball_kalman.py` | **The one surviving Kalman.** Delete the copies inside `main.py` and `ball_tracking.py`. |
| `src/association/data_association.py` | `track/ball_kalman.py` | Fold in — it is 21 lines and only used here. |
| `src/analytics/bounce_detector.py` (empty) | `ball/bounce.py` | Write fresh. Convention: **image y increases downward, bounce = local max in y.** |
| `src/analytics/pitch_length.py` | `ball/line_length.py` | **Rewrite** — current version thresholds normalised image `y`, which encodes the camera, not the pitch. |
| `src/analytics/release_speed.py` | — | **Drop** until calibration exists. This is the source of `release_speed_kmph: 409.9` (assigns `dist_px / time_sec` to a km/h field). |
| `src/detection/shot_classifier.py` | `detect/shot_classifier.py` | Keep; useful later for cut detection. |
| `src/events/*` | `fuse/events.py` | Consolidate — there are 5 overlapping boundary files. |

**Do not migrate:** `src/main.py`, `src/ball_tracking.py`, `src/test.py`, `src/test2.py` (four near-duplicate entry scripts), the empty `configs/*.yaml`, the empty `README.md`/`requirements.txt`, and every `*.mp4`/`*.json` at the repo root.

---

## Step 4 — Write the schema first

Before any pass runs, create `src/cricsense/schema/models.py` with the Pydantic models from [ARCHITECTURE.md §4](docs/ARCHITECTURE.md). Every pass validates on read and on write. This is what lets you run the ball pass without the player pass working.

---

## Step 5 — The three fixes that actually unblock identity

These are the bugs, not refactors. Do them before adding anything new.

**5a. `jersey_logged` is a class, not a set** — `player_registry.py` line 14:

```python
self.jersey_logged = set     # ← missing ()
```

`stable_id not in registry.jersey_logged` raises `TypeError` the moment a jersey locks, so the success path crashes.

**5b. `resolve_id` matches on role equality.** Current logic skips any candidate whose `final_role` differs from the incoming role. Role comes from the YOLO class name, which flips between `Batsman` and `Player_Generic` frame to frame, so each flip mints a fresh ID → **534 SIDs for ~13 people**. Replace the role check with IoU + centroid distance gating, and treat role as mutable metadata on the player, not part of the match key.

**5c. Role must come from geometry.** Implement `calibration/homography.py` (4 stump/crease correspondences → pitch metres), project each player's foot point, and assign role by pitch region. Delete `infer_role_from_class`.

---

## Step 6 — Config, entry point, cleanup

Move every hardcoded Windows path (`C:\cricket player train\...`, `C:\CricketSense-Ball\...` — they appear in most entry scripts across both repos) into `configs/default.yaml`:

```yaml
device: cuda:0
paths:
  player_weights: models/yolov8m_production4.pt
  ball_weights:   models/ball_stump_best.pt
  video_dir:      data/
  output_dir:     outputs/
```

Wire up `cli.py`:

```bash
cricsense calibrate --video data/match_01.mp4      # interactive 4-point picker
cricsense players   --video data/match_01.mp4
cricsense ball      --video data/match_01.mp4
cricsense fuse      --video-id match_01
cricsense render    --video-id match_01
cricsense live      --video data/match_01.mp4      # optional combined loop
```

Then:

```bash
rm -rf _src_ai _src_ball
git add . && git commit -m "Merge CricSense-AI and ball-tracking into one package"
```

Verify nothing large slipped through before pushing:

```bash
git count-objects -vH          # expect a few MB, not GB
```

---

## Step 7 — Verification checklist

- [ ] `pip install -r requirements.txt` succeeds in a fresh venv
- [ ] `grep -rn "C:\\\\" src/` returns nothing
- [ ] `cricsense players` runs end to end and writes valid `players.json`
- [ ] Stable ID count is within ~2× the true number of people on field (not 534)
- [ ] `cricsense ball` writes `ball.json` with no km/h field
- [ ] `cricsense fuse` produces at least one CONTACT event on a clip where the batter plays a shot
- [ ] `git count-objects -vH` shows no accidental weights or video
- [ ] `scripts/benchmark.py` fills the metrics table in ARCHITECTURE.md §8
