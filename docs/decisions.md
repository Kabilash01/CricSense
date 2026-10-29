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
found on 11 of the 14 (re-run 2026-10-14 after the multi-reference
calibration: the same 14 clips calibrate, all checked by eye on the painted creases; one more
delivery found, we_wide_19_1).
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
  884 frames, 6/6 deliveries. Matching every frame against every reference cost 2,693 s on
  this clip. Grouping references by camera position to skip some was tried and dropped: a pitch
  calibration is relative to the bowling direction, so the two ends' cameras get the same centre;
  the tracker test of a cut between ends failed. What is kept: within a shot a frame is matched
  only against the reference the last frame used, and all references after a cut (thumbnail
  grey-level change > 30; cuts measured 40-93, pans up to ~23), when the estimate jumps (corners
  > 80 px from the last frame's) or when that reference's inliers fall below half the last
  frame's (a cut between the ends' near-identical views: 453 -> 77). Bumrah's over 2,693 -> 446 s,
  the same 884 frames, lengths within 9 cm; India's 6 minutes 621 -> 351 s, lengths identical;
  `make check` all pass. Where the two differed most (zoomed frames, up to 62 px), the
  followed reference sat closer to the painted crease.
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

## Score-box font templates over the whole match

A full rescan (46 min) with the glyph reader, measured by `scripts/match_coverage.py`:

| | India | South Africa |
|---|---|---|
| before (Tesseract) | 118/121 ball counts, 114 single-ball steps | 115/121, 106 steps |
| first version | 118, 108 | **95**, 81 - rejected |
| fixed | 118, 114 | **117, 111** |

Final scores exact throughout (176-7, 169-8). The first version applied the box's score to the
layout's reading wherever the box was found; on a moved bar (the chase's compact bar) the
layout's overs field reads the wrong place, so a right score came with wrong overs and the
plausibility rules dropped those states (score-box readings 3,466 -> 425). The box's score now
replaces the layout's only when the box sits within 25 px of where the layout expects it (normal
centres 3-15 px off); a moved bar goes the score-box way, with the overs read beside the box
(test: the bar shifted -150 to +100 px). The rescan also found the score box on frames the old
reader missed, where the overs beside it picked up a stray digit ("3 2.40" for 2.4 overs): an
overs.balls number now wins over a bare one. `make check`: all three verified clips unchanged.

## Live mode re-measured (2026-10-15, GPU nearly free)

Same test (the 6-minute clip's first 150 s): 5/5 balls right, median 0.99 s after the bar settled
(was 1.8 s), radar 0.1 s. The report's wall time had included waiting at the end for shots
still being processed (215 s for 150 s of video), and its "late frames" were the ~8 s of model
loading counted against the first frames. The report now gives when playback ended
(`play_wall_seconds`: 150.0) and the worst lag (`reader_max_lag_s`: 0.16 s once the clock
starts after the models are loaded; it was 8.4 s at t = 0). Length and line: 4 shots, 1 bounce,
right (4.82 m vs 4.79 m offline) but 146 s late; 3 without a bounce. Still not live: per shot,
camera re-tracking and both ball models over the buffered frames take minutes.

## A second match: Australia v India, Super 8 (2026-10-20)

Whole-match scan with its own team sheet (`configs/matches/replay-super-8s-aus-vs-ind.yaml`):
India 205-5 (121/121 ball counts) and Australia 181-7 (119/121), the official result. Two overs
analysed end to end with `python main.py "<match video>" --overs 1:3 2:9`:
- Starc's over (1:3): 6, 6, 4, 6, 0, wide, 6 = 29, as bowled; 48/48 checks. It failed calibration
  at first - the main view appeared only in short bursts between replays - so automatic
  calibration now samples every 0.5 s (frames with pitch colour, kept only if the creases fit).
- Kuldeep's over (2:9): 1, 1, 0, 1, 1, wicket (Marsh, 87-2). The wicket ball had no outcome: the
  overs field was read digits-only, so "9 OVERS" became 90 and the state was thrown away; the
  field is now read with its letters and "OVERS" stripped.
- After a wicket the bar shows the incoming batter, so a dismissal is often seen only as a score.
  The batter who faced is now taken from the full bar before the ball when that bar showed the
  same score (Marsh, 8.6); the score engine still treats it as score-only (runs off the bat unknown).
- Calibration of a 7-minute over took 20-50 min (camera tracking: frames matching no reference are
  compared with all of them). Not yet fixed.

## Commentary on free APIs; shot reading off (2026-10-21)

Lightning AI stopped answering (401, then 404). The commentary lines now use Gemini's free tier
(`gemini-3.5-flash-lite`, one call per clip, ~4 s; `make llm-check` tests the key). 429/503 replies
are retried after the wait the API asks for; a used-up daily quota or an empty balance fails at once.

Shot reading was tried on the free vision models, on the 9 played shots of the two Super 8 overs:
- `gemini-3.8-flash`: timed out (2 min) on a 64-px test image. `gemini-3.6-flash`: 22 s per call,
  overloaded (503) during the run, and its free quota is 20 requests a day - one over, as each
  shot is read three times.
- `gemini-3.5-flash-lite`: "cover drive" for a push to a yorker, a horizontal-bat shot and a
  crouched leg-side shot.
- Groq `qwen/qwen3.8-27b` (4 frames tiled into one picture: it takes 3 images): three readings
  agreed on 3 of 9 shots; Marsh's dismissal - a sweep, caught at deep backward square leg - read
  as a flick all three times. Its description was half right (leg side, high finish, standing):
  the naming rules have no standing slog-sweep either.
Free-tier OpenAI does not exist (an account needs credit); Hugging Face gives free accounts no
inference credits; GitHub Models is retired. So shot reading is off by default (`vlm.enabled`); the
lines say only what was measured. To revisit: a hand-labelled answer key first, then a stronger
model and naming rules for the shots the rules miss.

The fact sheets had two bugs the LLM exposed: the non-striker was taken from the state after the
ball (after a single the ends have swapped, so it named the striker), and a pose reading of "left"
survived on a ball hit for six ("left alone for SIX!"). Both fixed, with tests.

## The broadcast's commentators out from under ours (2026-10-24)

Our commentary was mixed over the broadcast sound, ducked while our voice spoke; between our lines
the TV commentators were heard, two commentaries over one picture. Now the stadium sound is first
split by Demucs (htdemucs, MIT licence, local, ~80 MB) into voices and the rest, and only the rest
(crowd, bat, stumps) goes under our commentary (`broadcast_voices: remove`; `keep` and `mute` as
before / none). The stereo trick (the commentators sit in the middle: keep only L-R) was measured
first: L and R correlate 0.76 and the side signal is 9 dB below the middle - it would leave a thin,
phasey crowd, and any off-centre voice in.
Measured on 40 s of Bumrah's over (the voices stem of a second Demucs pass, before and after): the
commentators at -23.6 dB and speaking in 24 of the 40 s in the broadcast; after, -52 dB and 0.4 s,
with the rest of the sound 3 dB lower. 5 s on the GPU for the 40 s. On a whole over (final_0051_6min)
one pass was not enough: voice was still found in 25 s of the 290 s between our lines, only 7-14 dB
under the broadcast's (no bass there: commentators, not stadium music). A second pass over the
result left none, with the crowd 0.4 dB lower, for 16 s more per 6 minutes - so two passes. The
rendered overlay's sound matches the crowd track exactly (correlation 1.00), not the broadcast's.
The model is loaded from `models/demucs` (`make models`): demucs 4.1 fetches it from the Hugging
Face hub by default, which stalled at 0 bytes here, while Meta's server (the file the hub copies)
worked.

## Live plus: players named live, the delivery card, speed (2026-10-28)

`cricsense live-plus` (`make live-plus`) is live mode with a third worker: the players pass's
detector and tracker on the newest main-view frame, roles from where the feet stand (the offline
role assigner), names from the live score bar and the keeper from the match sheet. The picture is
shown 1 s behind the source so boxes are drawn on the frame they were found on (or 1-5 frames
later). Measured on Bumrah's over (5 min 20 s, RTX 5060 Laptop, nothing else on the GPU):
- Every frame matched to the reference views cost 120-330 ms. The camera is now followed by optical
  flow on a half-size copy (~10 ms), with a full match every 6 frames (`FlowCamera`). Against the
  offline per-frame calibration, where players stand: 0.25 m median, 0.64 m at the 90th percentile.
- The ball model analysing the last shot took the GPU from the players (their detector 360 ms a
  frame instead of 35): it now waits while the main view is on screen and catches up between balls.
- A shot's ball analysis took 25-66 s; now 8-15 s: FP16 (51 ms a frame at 1920 instead of 168, the
  same detections on 48 of 48 frames), the camera by flow, and the shot left 12 frames after a
  confirmed bounce (the fine-tuned model only from the release to there). Lengths against offline:
  within 0.1 m on 5 of 6 deliveries (the sixth 3.3 vs 4.2 m), estimates 137-144 km/h against radar
  137-142. Live mode (`cricsense live`) shares these changes.
- Result over the whole over: kept pace (327 s for 320 s), every outcome 0.5 s after the bar
  settled (6/6), radar 0.1 s, a bounce for each delivery 12-23 s after it, players named on the
  main view (striker, non-striker, bowler, keeper, umpire as on the bar).
- The score bar shows a ball's radar up to 4 s after its result (1.1: result 15.4 s, radar 19.7 s),
  so a reading within 8 s of a result is that ball's; the first card version gave it to the next.
Not live: the ball's path line (the delivery is found after its shot) and spoken commentary.
Replays can add a second bounce for a ball; the card prefers the confirmed, earlier one.

## Live plus on a whole match: calibrating from the creases (2026-10-29)

On the whole final, live-plus named no one: a camera reference matches the broadcast for minutes,
not hours (shadows and grass move). Measured on the first 5 minutes: the 51-minute clip's views
matched 1 of 291 main-camera frames, a 15-minute one 1, one from 2 minutes earlier 222-405 points
on the same frame (needs 80). Now:
- a shot starts on the pitch strip (pale-brown share of the frame: main view 0.31-0.54, fielding,
  crowd and outfield 0.00-0.09; `shows_strip`), not on a reference match; shots under 2 s are left
- references are tried once per shot; then the shot is calibrated from its painted creases
  (0.3-0.5 s), confirmed by a second strong calibration a few frames on within 20 px (as offline,
  one frame can be fooled by a batter on a crease), carried by optical flow, and added to the
  shared reference bank; at most 4 tries per shot (replays and pitch close-ups never calibrate)
Result, first 7 minutes on mains power: kept pace (424.6 s for 420), every ball's card, radar and
outcome; players named on 156 frames (was 0) - on about half the deliveries, for 1-2 s of the
run-up each (0.6 and 1.3 fully: striker, non-striker, bowler, keeper, umpire), none on 1.1 and 1.2;
roles can be wrong right after a shot, when everyone runs. On battery it fell behind (665 s).
An over with its own calibration (Bumrah's) is named throughout. Not solved: a reliable camera per
frame over a whole match in real time.
