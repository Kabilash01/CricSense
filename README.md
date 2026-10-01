# CricSense

Analysis of a TV cricket broadcast, ball by ball, with no manual input: who is batting and
bowling, where each ball pitched (length and line), how fast it was, what the batter did, what
the ball produced - turned into an event timeline, a checked match state, commentary (written
and spoken), an annotated video, and a database behind an HTTP API. A live mode follows a
stream as it plays.

Built and measured on the ICC Men's T20 World Cup 2024 final broadcast (India v South Africa).
Every number below is measured on clips that were not used to build or tune what they measure;
what did not work is recorded with its measurements in [docs/decisions.md](docs/decisions.md).

---

## What it does

| | How | Measured |
|---|---|---|
| **Pitch calibration** | Automatic: the regulation creases found in the picture, scored against the laws' dimensions; then the camera tracked frame to frame | 7 of 8 crease corners within 3.3 px of hand clicks; 14 of 20 new clips calibrated, none wrongly |
| **Players** | Detection, tracking, roles from pitch geometry and movement; names from the score bar; teams from the match sheet | 51 identities for 51 appearances on 12 shots |
| **Ball** | Two detectors (one fine-tuned on 94 hand-reviewed frames of this broadcast); delivery, bounce, length, line | 10/10 deliveries, 9 bounces; bounce 4.77 m vs 4.8 m by hand |
| **Speed** | The broadcaster's radar from the score bar; our own time-of-flight estimate shown as "≈" | Estimate vs radar: 6% median error (3 balls) |
| **Outcomes** | Runs, boundaries, wickets, wides, no-balls from the score bar's changes | 15/15 balls on the two test clips; both innings' final scores read exactly over the whole match |
| **Score engine** | A ball-and-over state machine (strike rotation, extras, figures), checked against the bar | 88/88 and 40/40 checks agree |
| **Batter** | Pose: footwork on the calibrated pitch, swing (played / left), batting hand | Swing right on 9/9 balls; hand 10/10 |
| **Shot type** | A vision model describes the posture three times; textbook rules name the shot when two agree | Labelled "AI read"; 5 of 8 shots agreed |
| **Commentary** | An LLM given only each ball's facts, every line fact-checked (numbers, names, events); templates as fallback | |
| **Voice** | Local Kokoro TTS, mixed over the stadium sound | ~0.4 GB memory, no API key |
| **Overlay** | Ball path held on the pitch through camera moves, bounce, delivery card, pitch map, wagon wheel, captions | |
| **Storage / API** | PostgreSQL + object storage; FastAPI: upload a video, read deliveries, events, scorecard, commentary, files | |
| **Live** | Score, outcomes, boundaries, wickets and radar on a stream in real time | 5/5 balls, median 1.8 s after the bar settled |

Not done, because the broadcast does not show it well enough (measurements in
[docs/decisions.md](docs/decisions.md)): runs counted from the batters running, wides by ball
position, front-foot no-balls, a fielder map, knee-down for sweeps. Length and line in live mode
need a free GPU.

---

## Install

```bash
conda create -n ball python=3.11 && conda activate ball
pip install -r requirements.txt          # torch 2.14 (CUDA 13) runs on RTX 50-series
sudo pacman -S tesseract ffmpeg postgresql   # or your distribution's packages
```

Models go in `models/` (not in git): `ball_best.pt`, `ball_ft.pt`, `bat_best.pt`, `yolov8m.pt`,
`yolo11x-pose.pt`, and `tts/kokoro-v1.0.int8.onnx` + `tts/voices-v1.0.bin`
([kokoro-onnx model files](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0)).
Optional keys in `.env` (gitignored): `LIGHTNING_API_KEY` (commentary and shot reading through
Lightning AI) or `GEMINI_API_KEY`; without one, commentary uses templates.

## Run

```bash
cricsense run --video data/clip.mp4              # every step; existing outputs are skipped
cricsense run --video data/clip.mp4 --voice      # ... and speak the commentary
```

Results in `outputs/clip/`: `overlay.mp4`, `pitch_map.png`, `wagon_wheel.png`, `events.json`,
`match.json`, `commentary.json`, `ball.json`, `players.json`, `pose.json`.

Steps one at a time: `calibrate --auto`, `scoreboard`, `players`, `ball`, `fuse`, `pose`,
`commentary`, `voice`, `render` (`cricsense <step> --help`).

```bash
scripts/pg.sh start                              # a project-local PostgreSQL (no root)
cricsense publish --video-id clip                # results into the database and object storage
cricsense serve --port 8000                      # API: http://localhost:8000/docs
cricsense live --video data/clip.mp4 --calibration configs/calibration/<clip>.yaml --out live.mp4
```

## For a new match

1. `configs/match.yaml`: the two squads as the score bar prints them, and the keepers.
2. `configs/scoreboard.yaml`: the broadcaster's score-bar layout (one file per graphics package).
3. Run. Calibration is automatic; for short clips `calibrate --auto --cameras-from <another clip of the match>.yaml` helps.
The ball detector was fine-tuned on this broadcast; another may need a short labelling round
(`scripts/prepare_ball_labelling.py`, `scripts/label_ball.py`, `scripts/finetune_ball.py`).

---

## Speed

`scripts/benchmark.py`, 6-minute clip, RTX 5060 Laptop + Ryzen 9 8940HX (another job on the GPU):
every step runs faster than real time except camera tracking (0.14x; 1.4x faster since by
skipping frames with no pitch in view). Full table in [docs/decisions.md](docs/decisions.md).

## Layout

```
src/cricsense/   calibration/ (auto, camera) identity/ (score bar, names) ball/ fuse/ (events,
                 outcome, score engine) pose/ commentary/ render/ live/ store/ api/ passes/ cli.py
configs/         every threshold and path; no absolute paths in source
scripts/         benchmark, evaluation, labelling and fine-tuning, clip cutting, local PostgreSQL
docs/            ARCHITECTURE.md, decisions.md (what was measured and decided), adr/, history/
tests/           172 tests: synthetic cameras and scenes, real fixtures, API on SQLite
```
