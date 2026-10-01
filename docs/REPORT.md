# CricSense: ball-by-ball analysis of a cricket broadcast

Technical report. Everything quantitative here is measured, on clips not used to build or tune
what is measured unless said otherwise; the detail behind each decision is in
[decisions.md](decisions.md).

## 1. Problem

Given a TV broadcast of a cricket match - one video, no access to the broadcaster's cameras,
tracking data or scorers - produce for every ball: who bowled and who faced, where it pitched,
how fast it was, what the batter did, and what it produced; keep a match state that follows
the laws; and present it as an annotated video, commentary, and data an application can query.
The work started from two earlier repositories (player detection; ball tracking), whose
claims were checked first: several did not hold (a 409.9 km/h "speed" no code produced; a
player model whose weights were lost and which, retrained, found nobody in this broadcast).

**Data.** The ICC Men's T20 World Cup 2024 final (India v South Africa), a 5 h 21 min, 25 fps,
1080p broadcast. Evaluation clips: overs 11.1-12.4 of India's innings (6 min) and 7.1-7.5 of
South Africa's (3 min, never used in development); a benchmark of 20 clips cut at random from
the whole match around the balls the score bar identified.

## 2. System

Independent passes joined by JSON files with strict schemas (`src/cricsense/schema`); each
runs, fails and is timed alone, and `cricsense run` chains them.

| Pass | What it does |
|---|---|
| `calibrate` | Finds the pitch's camera views from the painted creases and tracks every frame (§3.1) |
| `scoreboard` | Reads the broadcast score bar every 0.5 s (§3.5) |
| `players` | Detects and tracks people; roles from pitch geometry and movement; names from the bar (§3.2) |
| `ball` | Two ball detectors; delivery, bounce, length, line, speed (§3.3) |
| `fuse` | Events (release, bounce, contact, runs, boundary, wicket, wide, no-ball) and the score engine (§3.5) |
| `pose` | The striker's footwork, swing and batting hand (§3.4) |
| `commentary` / `voice` | Fact-checked commentary, shot reading, local speech (§3.6) |
| `render` | The overlay video, pitch map, wagon wheel (§3.7) |
| `publish` / `serve` | PostgreSQL + object storage; FastAPI (§3.8) |
| `live` | The above, degraded, on a stream as it plays (§3.9) |

## 3. Methods and results

### 3.1 Calibration

The main camera stands behind the bowler, turns and zooms but does not move. A pitch's
painted creases are fixed by the laws, so calibration needs no clicks: white-line candidates
(top-hat filter, pitch-coloured ground on both sides) give crease-corner hypotheses, each scored
by how much of the regulation creases lands on paint plus whether the stumps stand where the
middle-stump base projects; homographies no real camera could produce are rejected. Frames are
grouped by where their calibration places the camera, and a view needs two strong agreeing
frames (or three). The tracker then follows each frame (ORB, RANSAC, outlier-robust
smoothing), skipping frames with no pitch in view.

| | Result |
|---|---|
| Against a hand-clicked frame | 7 of 8 crease corners within 3.3 px |
| Where the hand calibration had drifted (bowler's end) | the automatic one sits on the paint |
| Second innings (never calibrated by hand) | calibrated, creases on the paint on all 5 deliveries |
| 20 new clips | 14 calibrated, none wrongly (precision first: a wrong calibration corrupts everything after it) |
| A 3D camera per frame (fixed centre, zoom and rotation fitted) | heights to ~2-3% |

### 3.2 Players

A COCO person detector at 1280 px (the retrained cricket-player model, 0.895 mAP50 on a clean
test split, finds almost nobody in this broadcast's views - domain shift), BoT-SORT with the
calibrated camera motion as global motion compensation, a registry that relinks tracks on the
pitch. Roles come from pitch geometry and movement, names from the score bar (striker by the
bat icon), teams and the keeper from the match sheet. 51 identities for 51 player appearances
over 12 delivery shots.

### 3.3 Ball, bounce, speed

Two detectors: the original model finds each delivery; one fine-tuned on 94 frames of this
broadcast, reviewed by hand in a purpose-built local labeller, gives the track (false balls on
crease marks 131 -> 20). A delivery is a physically constrained path from a release anchor;
the bounce is where the ball's ground point turns.

| | Result (6-minute clip) |
|---|---|
| Deliveries found | 10/10 (either model alone: 10 or 9) |
| Bounces | 9 (8 or 6 alone); hand-checked: 4.77 m vs 4.8 m |
| Speed | the broadcaster's radar, read from the bar, when shown (6 of 10 balls) |
| Our estimate ("≈", time of flight release -> bounce) | 6% median error against radar (3 balls; worst 14%) |

The original design's speed - pixel displacement through the ground homography, with a Kalman
filter - was measured at 1-72 km/h against radar 81-146: an airborne ball is not on the ground
plane, and from behind the bowler its sight line barely moves.

### 3.4 The batter

Pose (YOLO11x-pose on crops around the striker) on the calibrated pitch: footwork only where
the feet are really visible (the camera zooms in after release); the swing from wrist speed
relative to the hips, so camera motion cancels - "played" on all 8 balls with runs off the bat
and "left" on the one not hit; the batting hand from where the feet stand at the stance - 10/10.

### 3.5 Outcomes and the score engine

Outcomes come from the score bar's changes, which must be stable (seen twice), plausible (never
backwards, never faster than a ball per 15 s) and recognised (squad names, surname matching
for prefixes like "de KOCK"). Where the layout's rectangles miss (the bar slides; the chase's
compact bar), the score box is found by colour. A ball-and-over state machine then keeps the
score, strike, extras and figures by the laws and is checked against the bar after every ball.

| | Result |
|---|---|
| Outcomes on the two test clips | 15/15 (incl. a six and a four) |
| Score engine checks | 88/88 and 40/40 |
| Whole match | both innings' final scores exact (176-7, 169-8); India 118/121 ball counts, South Africa 89/121 before the score-box reader |

The score and overs checks are consistency checks (the outcomes come from the same bar); strike
rotation, balls faced and bowler figures are independently tested.

### 3.6 Commentary, shot type, voice

An LLM (Lightning AI / Gemini, any OpenAI-compatible API) sees only each ball's facts; every
line is checked - every number must be in the facts, only the players involved named, no
boundary or wicket the facts lack, a word budget to fit the gap - and replaced by a template
line if it fails. Shot type: asked for a shot name, vision models guessed ("sweep" for almost
everything); asked only to describe posture, they were consistent, so the shot is named by
textbook rules from the description, read three times and kept when two agree (5 of 8 shots).
Speech: local Kokoro (int8 ONNX, ~0.4 GB, no key), pace set by the moment, mixed over the
crowd with ducking.

### 3.7 Overlay

The ball's path drawn on the pitch and held there while the camera moves (the camera's centre
is fixed, so one homography carries any point between frames), the bounce, a delivery card
(length, line, speed and its source, shot, footwork, outcome), banners, a pitch map, a wagon
wheel (shot angle by type, mirrored for left-handers; labelled approximate), captions.

### 3.8 Storage and API

PostgreSQL (SQLite in tests) through SQLAlchemy: matches, videos, one row per ball joining every
pass, events, score states, commentary, files; object storage local or S3-compatible. FastAPI:
upload a video (the pipeline runs as a job and publishes), read deliveries, events, scorecard,
commentary, download the overlay and images.

### 3.9 Live mode

A reader plays the source at its pace; workers take only the frames they can keep up with. On
150 s played in real time (151.5 s wall): each ball's outcome, boundaries, wickets and the radar
speed live - 5/5 balls right, median 1.8 s after the bar settled. Length and line are found per
shot from a frame buffer in a background thread, but beside another job on the GPU a shot takes
over a minute: not claimed as live.

## 4. Speed

6-minute clip, RTX 5060 Laptop GPU, Ryzen 9 8940HX, another training job on the GPU:

| step | s | x real time |
|---|---|---|
| calibrate (camera tracking) | 2529 (now 1.4x faster) | 0.14 |
| scoreboard | 149 | 2.4 |
| players | 64 | 5.6 |
| ball | 305 | 1.2 |
| fuse | 40 | 9.1 |
| pose | 27 | 13 |
| commentary | 107 | 3.4 |
| voice | 106 | 3.4 |
| render | 102 | 3.5 |

Everything but camera tracking runs faster than real time.

## 5. What did not work, and why

Each measured before being dropped ([decisions.md](decisions.md)):
- **Runs from the batters running**: the broadcast shows the pitch for at most 1.8 s unbroken
  after the shot on 14 of 15 balls; a run takes 3-3.5 s.
- **Wides by ball position**: on the match's own 5 wides (found by the score bar), the lateral
  measurement near the batter was not trustworthy (legal balls measured 1.3-2.75 m wide).
- **Front-foot no-ball**: 3.5 cm per pixel at the bowler's crease from this camera; margins are
  1-5 cm.
- **Fielder map**: the main view shows five people at the delivery; no frame shows the field.
- **Knee-down (sweeps)** from in front; **release timed from the bowler's arm** for speed (kept
  only as a fallback for finding deliveries); **half-resolution camera tracking** (faster, less
  accurate); **layout-free score-bar readers** (2-3 of 120 frames).

## 6. Limits and next steps

- Generalising to other broadcasts needs a score-bar layout per graphics package and may need a
  short labelling round for the ball detector; calibration and the rest are automatic.
- Deliveries are missed when the ball is not detected near the release (4 of 14 calibrated
  benchmark clips): more labelled frames of the early flight would help most.
- Shot type is an AI reading, not a measurement; a labelled answer key would let it be scored.
- Live length and line need a free GPU or faster inference (FP16/TensorRT, a crop on the pitch).
