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
