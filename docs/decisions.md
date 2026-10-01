# Measured decisions

Things tried on the broadcast and kept, changed or dropped because of what the measurements
showed. Evaluation clip: `final_0051_6min` (ICC T20 WC 2024 final, overs 11.1-12.4), not used
for any training or tuning of thresholds unless stated.

## Ball speed: broadcaster radar first, our estimate only with a clean release and bounce

- Pixel displacement through the ground homography (the original design): 1-72 km/h against
  radar 81-146. An airborne ball is not on the ground plane; seen from behind the bowler its
  sight line barely moves.
- Time of flight, release to bounce, with the fine-tuned ball model's track: median error 6%
  (3 radar balls, worst 14%). Shown as "≈ ... estimate", only with a confirmed bounce.
  The worst case is pace: about 8 frames from release to bounce at 25 fps, so one frame is
  ~13% of the speed (spinners: ~15 frames, ~7%), and a fast ball's first frames of flight are
  often missed, so its release is seen late. Bumrah's over added 5 radar balls: 3 estimates came
  out at 171-198 km/h. With all 10: the 4 above 161.3 km/h (the fastest ball ever recorded) were
  15-43% fast; the 6 below, 0-9% off (median 6%). Estimates above 161.3 km/h are withheld.

## Two ball models

`ball_best.pt` finds each delivery (10/10); `ball_ft.pt`, fine-tuned on 94 hand-reviewed frames
of this broadcast, gives the track (131 -> 20 false balls on crease marks). Together: 10/10
deliveries, 9 bounces (8 and 6 alone). See `configs/ball.yaml`.

## Front-foot no-ball: dropped as a measurement

- The bowler's end is at the bottom of a telephoto view from far behind: 3.5 cm per pixel
  along the pitch at the popping crease; no-ball margins are 1-5 cm.
- At 25 fps the front foot lands between frames; in several "landing" frames it was still in
  the air, and once the bowler's box had switched to the keeper.
- The calibration at the bowler's end was 30-50 px off the painted crease.
No-balls come from the score bar (the NO_BALL event); TV umpires use side-on, high-frame-rate
crease cameras for this.

## Batter pose: footwork and swing kept, knee-down dropped

- The broadcast zooms in on the batter after release and the feet often leave the picture;
  the pose model then puts ankles on the shins. Feet are only trusted when visible, inside the
  picture and below the knee; a stride over 1.5 m is rejected as a misread.
- Swing (wrist speed relative to the hips): "played" on all 8 balls with runs off the bat,
  "left" on the one ball not hit (12.3).
- Knee on the ground: false on an upright Kohli (12.4). From in front, a bent knee over a
  planted foot looks the same as a knee on the ground. Dropped.

## Shot type: the VLM describes, rules name

Asked for a shot name, both models tried called almost every shot a "sweep" - including
upright drives they described correctly when asked what they saw. The VLM now only reports
posture, foot, bat finish and ball side; the shot is named by textbook rules
(`commentary/vlm.py: name_shot`), with the measured footwork from pose taking precedence.
Shown as "(AI read)". Not yet checked against a hand-labelled answer key.

## Calibration: automatic from the painted creases

Every pitch has the same creases, so the main view is calibrated without clicks
(`calibrate --auto`): hypotheses from white lines, scored by how much of the regulation
creases lands on paint plus the stumps' position, impossible cameras rejected, the best frame
of each camera view kept as its reference. Dev frame: 7 of 8 corners within 3.3 px of the
hand clicks; where the hand-clicked view had drifted (bowler's end, frame 6563) the automatic
one sits on the paint; the second innings, never calibrated before, calibrates on all 5
deliveries of the test clip.

## Fielder map: not attempted

At the moment of delivery the main view shows five people (bowler, batters, keeper, umpire)
on all 10 balls of the evaluation clip. Fielders appear only in follow and wide shots that
show one to three of them, from cameras at other angles where the creases are a few pixels
long. No frame shows the field setting, so a fielder map from the broadcast would be mostly
empty; it needs a fixed wide camera or the broadcaster's own tracking data.

## Runs by running between the wickets: not measurable from the broadcast

The broadcast shows the pitch for at most 1.8 s unbroken after the shot on 14 of 15 balls
(one ball 3.8 s); a run takes about 3-3.5 s, two about 7 s. The picture cuts to the chasing
fielder, the ball or a close-up, so the running is mostly off screen. Runs, boundaries and
wickets are therefore read from the score bar (labelled so), and the score engine's checks on
score and overs are consistency checks, not independent measurements; strike rotation,
balls faced and bowler figures are the genuinely tested parts.

## Wides by vision: measured, not good enough to ship

Evaluation set: the 5 wides of India's innings (found by the score bar over the whole
broadcast) and 15 random legal balls, each cut as a clip and run through the vision passes
with no clicks (`scripts/wide_eval.py`, results in outputs/wide_eval.json).
- Pipeline on new clips: calibrated 16/20 (one clip has no pitch view), delivery found 14/16.
- Only 3 of the 5 wides got a lateral measurement, and the measurement is not trustworthy:
  three legal balls came out 1.3, 1.7 and 2.75 m from the middle-stump line (a wide by any
  rule), one wide 0.3 m outside off, one wide with bat contact. The track near the batter is
  sometimes another object (a throw, a replay), and the T20 wide also depends on where the
  batter moved to, which the broadcast view does not show well.
Wides stay read from the score bar. What it would take: the ball tracked reliably past the
batter (more labelled frames of this part of the flight; 50 fps footage), the batter's final
position, and more wides to test on (5 per innings; several matches).

## Score bar over a whole match

- A stable misread that jumps ahead ("1.0" read as "10.0") used to be accepted and then block
  every genuine state as "going backwards". States must now be reachable in the time elapsed
  (at most one ball per 15 s, 8 runs per ball). Full final: India 67 -> 118 of 121 ball
  counts seen, South Africa 24 -> 89; both final scores read exactly (176-7, 169-8).
- The chase shows other bar variants (score moved left beside "NEED 26 FROM 24 BALLS",
  comparison panels, full-width FOUR/WICKET banners). Two layout-free readers were tried on
  120 missed chase frames: OCR of the whole bar strip recovered the score on 2, locating the
  score box by its colour on 3 (several panels share the colour). Not adopted; a broadcaster's
  bar variants need their own layout entries.

## Pipeline on new clips (benchmark: 20 random clips of the full final)

`scripts/wide_eval.py --cameras-from <a calibrated clip of the match>` runs the vision passes
on each clip with no clicks. Current: calibrated 14/20 with no wrong pitch (precision first:
a wrong calibration silently corrupts everything after it; a missing one is visible); delivery
found on 10 of the 14.
- The 6 calibration misses: a clip with one pitch frame, a dugout/advert clip with none, and
  views where only one frame is strong (a second agreeing frame or a known camera is needed).
- The 4 delivery misses: no ball detection near the release - the candidates there sit at
  fixed spots (stump tops, crease marks), the ball itself is blurred against the bowler or
  the main camera cuts in after release. Widening the release zone would let the static
  objects in. Fixes: the release time from the bowler's arm (pose) and labelled frames of the
  first part of the flight for the ball model.

## Release from the bowler's arm: built as a fallback

The bowling hand peaks ~0.3 body heights above the shoulders and drops below them within 1-2
frames of the release (`pose/bowler.py`): on the dev clip it agrees with the ball-track release
to within one frame on 7 of 8 deliveries. Used for every release, it moved the speed check
against radar from 6% to 22% median error (the speed model was validated on ball-anchored
releases) and a swing in the bound once beat the real release; so it is only a fallback, tried
when no ball detection anchors a release, and such deliveries get no speed estimate. On the
20-clip benchmark it recovers none of the 4 missed deliveries - their causes are upstream: no
ball track after the swing (13_3), no bowler found in a wide-camera view (13_6), the bowler too
small in the wide view for a clean arm signal (13_1).

## Benchmark (scripts/benchmark.py), 6-minute clip, RTX 5060 Laptop + Ryzen 9 8940HX

calibrate 2529 s (0.14x real time: feature matching at full resolution on every frame,
including the 93% that are replays and close-ups), scoreboard 149 s (2.4x), players 64 s
(5.6x), ball 305 s (1.2x), fuse 40 s (9x), pose 27 s (13x), commentary 107 s (3.4x), voice
106 s (3.4x), render 102 s (3.5x); all 57 min (0.11x). Another training job shared the GPU.
Every step but calibration runs faster than real time.

## Shot type readings are not stable between runs

With the LLM cache cleared (for the benchmark) the VLM's posture readings changed: Axar's six
went from "slog sweep" to "cover drive", Kohli's 11.2 from "back-foot punch" to "slog sweep".
The cache had hidden this. Shot type stays labelled "AI read"; to make it trustworthy, read
each shot several times and keep only answers that agree.

## Live mode (`cricsense live`): score and outcomes live; length and line need a free GPU

A reader thread plays the source at its own pace and offers frames to workers that drop what
they cannot keep up with. 150 s of the dev clip played in real time (151.5 s), with another
training job on the GPU:
- score, each ball's outcome, boundaries, wickets, radar: all 5 balls right (incl. the six),
  emitted a median 1.8 s after the score bar settled (8 s for the first, warming up); radar
  median 0.8 s.
- length and line: main-view shots are noticed every few frames and processed when they end
  (re-tracked in order from a frame buffer, both ball models at full resolution, in their own
  thread so the next shot is not missed). 4 shots noticed; but beside another job on the GPU a
  shot takes over a minute (memory falls back to one frame at a time, then the CPU), and the
  two processed had no bounce (offline: one inferred, one none). Needs a free GPU or faster
  inference (FP16/TensorRT, a crop around the pitch); not claimed as working live.

## Score box found by colour: South Africa's chase, before and after

The score bar slides ~20 px sideways or up at times, and the chase shows a compact bar; the
layout's fixed rectangles then cut the digits. Finding the pink score box and reading its
digits (and the overs to its right) recovered 96 of 120 missed chase frames in a test, and over
South Africa's whole innings: ball counts seen 90 -> 115 of 121, consecutive single-ball steps
76 -> 106, final score 169-8 read exactly. 3474 of 8210 readings came from the score box. The
scan of the innings took ~50 min once frames with no bar on screen skip the search (a first
version, run beside other jobs, was stopped after 7 h).

## Pace: Bumrah's over 2 of South Africa's chase (unseen clip, 5 min)

Added so the demo and the measurements cover pace, not only spin in the chase. It exposed three
faults, each fixed and re-measured on all three test clips (no change on the other two: 88/88
and 40/40 score checks as before):

- **Calibration references.** One reference frame per camera: the camera zooms hard during a
  delivery, so feature matching found only 206 of the clip's frames and missed the wicket ball.
  Each camera now keeps up to six strongly calibrated (>= 0.78) frames at least 10 s apart:
  884 frames, 6/6 deliveries. Cost: every frame is matched against every reference (~2.3x the
  matching time). Matching only the last-used reference was tried and dropped: a pitch
  calibration is relative to the bowling direction, so the two ends' cameras get the same centre
  and cannot be told apart by position; the tracker test of a cut between ends failed.
- **Score digits.** Tesseract read the score box's "11-1" as "1-1", "11", "111" or "-1" under
  every page mode and scaling tried, freezing the over at 7-1. The box's digits are now read glyph
  by glyph against templates of the font (`configs/scoreboard_digits.npz`, harvested by
  `scripts/harvest_digits.py` from 3,453 unambiguous readings; leave-one-out 250/250). Over 2:
  54/54 checks, the wicket and the four right.
- **A second "delivery" 2 s after a real one** (the ball in play, seen in a wide view): two
  deliveries cannot be under 10 s apart; the later is dropped. Radar now goes to the last
  delivery before the reading appears, not the last tracked shot (it had moved 142.4 km/h to
  the false one). An aborted run-up (no ball recorded by the bar) still shows as a delivery with
  no outcome ("no evidence").
