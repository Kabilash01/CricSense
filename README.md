# 🏏 CricSense

Cricket video analysis: player tracking with role and identity, ball trajectory with bounce and length, and a fused event timeline with an annotated overlay.

Built as an internship project. Player detector: **YOLOv8m, mAP50 0.912 / mAP50-95 0.598** (65 epochs).

---

## Quick start

```bash
git clone <this-repo> && cd cricsense
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Drop your weights into `models/` and a clip into `data/`, then:

```bash
# 1. calibrate once per camera setup — click the 4 stump/crease points
cricsense calibrate --video data/match_01.mp4

# 2. run the two GPU passes
cricsense players --video data/match_01.mp4
cricsense ball    --video data/match_01.mp4

# 3. fuse into events and render the overlay
cricsense fuse   --video-id match_01
cricsense render --video-id match_01
```

Results land in `outputs/match_01/`: `players.json`, `ball.json`, `events.json`, `overlay.mp4`, `pitch_map.png`.

---

## How it works

Four passes over one video, joined by JSON files rather than shared memory. Each pass runs alone, fails alone, and is benchmarked alone.

| Pass | Device | Produces |
|------|--------|----------|
| `players` | GPU | Tracks, stable IDs, roles from pitch geometry, identities from the scoreboard |
| `ball` | GPU | Ball trajectory, bounce frame, line and length in pitch coordinates |
| `fuse` | CPU | Ball–bat contact, boundaries, wickets → event timeline |
| `render` | CPU | Overlay video and 2D pitch map |

Full rationale in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the merge decision is recorded in [ADR-001](docs/adr/ADR-001-repository-merge-and-runtime-model.md).

`cricsense live --video <path>` runs passes 1–3 in a single loop. It is a demo path — expect roughly 20–25 FPS on a laptop RTX for two detectors at 640px.

---

## Configuration

Everything path- or threshold-related lives in `configs/`. No absolute paths in source.

| File | Purpose |
|------|---------|
| `configs/default.yaml` | Device, weight paths, I/O directories |
| `configs/players.yaml` | Detector confidence, tracker params, size gates |
| `configs/ball.yaml` | Ball confidence, area/ratio filters, Kalman noise |
| `configs/calibration/<video_id>.yaml` | Homography matrix, written by `calibrate` |

---

## What this system does *not* claim

Stated up front, deliberately:

- **No ball speed in km/h.** A single uncalibrated broadcast camera cannot measure it. The system reports line and length, which are geometrically recoverable from the homography.
- **No true 3D trajectory.** Single camera; the trajectory is a plane-constrained estimate.
- **No authoritative scoring.** Scoreboard OCR is used for player identity, not to adjudicate runs or wickets.
- **Broadcast cuts are not yet handled.** Replays and camera changes will break tracking until shot-boundary detection is added.

---

## Repo layout

```
src/cricsense/
  cli.py          config.py       schema/       # entry point, config, contracts
  io/             calibration/                  # video I/O, pixel ↔ pitch
  detect/         track/                        # YOLO detectors, ByteTrack + Kalman
  identity/                                     # stable IDs, roles, scoreboard + jersey OCR
  ball/           fuse/           render/       # trajectory, events, overlay
configs/  models/  data/  outputs/  docs/  scripts/  tests/
```

`models/`, `data/`, and `outputs/` are gitignored — weights and video never enter version control.

---

## Development

```bash
pytest tests/                    # unit tests
python scripts/benchmark.py      # per-stage metrics table
```

Building this repo from the two original repos? See [SETUP.md](SETUP.md) for the file-by-file migration.

## History

Merged from `CricSense-AI` (player/identity) and `CricSense-ball-tracking-and-Tracjectory-` (ball). See ADR-001 for why.
