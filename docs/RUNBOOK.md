# CricSense runbook

How to run, check and extend CricSense yourself. Every task is one `make` command (`make help`
lists them); each section says what to run, how long it takes, how to tell if it worked, and
what to do when it did not. The commands run in the conda env `ball`; the Makefile finds it.

---

## 0. Before anything: is the machine ready?

```bash
make doctor     # programs, packages, models, GPU, API keys - says READY or what is missing
make test       # ~30 s; all must pass
```

If `make doctor` says MISS: the line under it says how to fix it. Models are not in git: copy
them into `models/` (README: Install). API keys go in `.env` (never commit it; never paste a key
into a chat or a commit). Without a key the commentary uses templates - everything else works.

The GPU: ball, players and pose use it. Another training job on the same GPU is fine but slower;
if a step dies with "CUDA out of memory", run it again when the GPU is freer.

---

## 1. Analyse a piece of the match

```bash
make clip START=6840 END=7160 CLIP=bumrah_o2    # cut it from the match video (seconds or HH:MM:SS)
make run CLIP=bumrah_o2                         # everything, with spoken commentary (~10-30 min per 5 min)
```

Results in `outputs/bumrah_o2/`: `overlay.mp4` (the annotated video), `match.json` (the score
after every ball), `events.json`, `pitch_map.png`, `wagon_wheel.png`, `commentary.json`.

A clip should start a few seconds before a ball and can be any length; 3-6 minutes (an over)
is a good unit. `make clip` records where the clip came from (`configs/clips.yaml`), so the
overlay gets the stadium sound.

**Did it work?** Look at three things:

1. `outputs/<clip>/calibration_check.png` - green lines on the white painted creases at both
   ends. If they are anywhere else, stop: see "Calibration is wrong" below.
2. The `fuse` summary printed at the end (or `python -m cricsense fuse --video-id <clip>`): one
   line per ball, and `all checks: N/N agree with the score bar`. Every check should agree.
3. Watch `overlay.mp4`: the ball path on the pitch, the card's length/line/speed, the score.

A step that already has its output is skipped. To redo steps:
`python -m cricsense run --video data/<clip>.mp4 --fresh ball fuse render` (or `--fresh all`).

---

## 2. Check that nothing broke (after any code or config change)

```bash
make test
make check          # re-runs the hand-verified clips and compares ball by ball (~10 min)
make check-quick    # the same comparison without re-running (seconds)
```

`make check` prints PASS/FAIL per clip and, on a failure, exactly which ball changed and how.
The answer key is `configs/golden.yaml`: every ball of the three verified clips (India overs
11.1-12.4, Kuldeep's over, Bumrah's over), checked by a person against the broadcast.

- **FAIL with a ball that got worse**: the change broke something - undo it or fix it.
- **FAIL with "balls not in the answer key"** or more deliveries: possibly an improvement. Watch
  those balls in the broadcast; only if the new answer is right, `make save-golden` and commit
  `configs/golden.yaml` with the change.

Never `make save-golden` without looking: it makes whatever the pipeline says the new truth.

---

## 3. The benchmark: does it work on clips it has never seen?

```bash
rm -rf outputs/we_* configs/calibration/we_*.yaml    # to run fresh (the clips in data/ stay)
make benchmark                                       # ~20 min
```

Prints one line per clip (`ok`, `no delivery found`, `pipeline failed at calibrate`), then
writes `outputs/calibration_sheet.jpg`. **Open the sheet and check every tile by eye**: the
green lines must lie on the painted creases. Current: 14 of 20 calibrated, none wrongly; a
delivery found in 11 of the 14. A wrong calibration is worse than none - it silently corrupts
everything after it.

---

## 4. The whole match: how much of the score was read

```bash
make scan        # reads the score bar over the whole broadcast (~2-3 h); writes outputs/<match>/scoreboard.json
make coverage    # ball counts seen per innings, and the final scores read
```

Compare the final scores with the official scorecard - they must match exactly (this final:
India 176-7, South Africa 169-8). Ball counts seen: 118/121 and 117/121 at present. To scan a
part only: `python -m cricsense scoreboard --video "<match>" --start 6500 --end 12800 --out-name sa.json`.

---

## 5. The demo video

1. Run each clip you want in it with voice (`make run CLIP=...`).
2. Edit `configs/demo.yaml`: the title cards' text and the clips, in order.
3. `make demo` - writes `outputs/demo/cricsense_demo.mp4` (~5 min for ~15 min of video).

---

## 6. A new match from the same broadcaster (ICC 2024 graphics)

Each match has its own team sheet, `configs/matches/<video file name>.yaml`; a clip finds it
through `configs/clips.yaml` (where `make clip` records the broadcast it was cut from), so
matches never overwrite each other.

1. Put the video in `Cricket match/`, e.g. `Cricket match/new-match.mp4`.
2. Copy a sheet: `cp configs/matches/replay-super-8s-aus-vs-ind.yaml configs/matches/new-match.yaml`
   and set both squads **exactly as the score bar prints the names** (e.g. "DE KOCK"), the team
   names and codes, and the wicketkeepers. Not sure of the spelling? Open a few frames with the
   bar on screen and copy it. A missing sheet is reported in the log ("no match sheet for ...").
3. `make clip MATCH="Cricket match/new-match.mp4" START=... END=... CLIP=...` and
   `make run CLIP=...` as in section 1. Whole match's scores: `make scan MATCH=...` then
   `make coverage MATCH=...`. Calibration is automatic. For a short clip
   (under a minute) that fails to calibrate, use a good calibration of a longer clip of the same
   match: `python -m cricsense calibrate --video data/<clip>.mp4 --auto --cameras-from configs/calibration/<longer clip>.yaml`.

## 7. A new broadcaster (different score-bar graphics)

1. `configs/scoreboard.yaml` - copy it, then set every field's rectangle `[x0, y0, x1, y1]` in
   pixels of a 1920x1080 frame: open a frame (`ffmpeg -ss 600 -i match.mp4 -frames:v 1 f.png`)
   in an image viewer that shows pixel coordinates and read off each field (batters, score,
   overs, bowler, the radar speed line), the score box colour (`anchor.hue`) and the bat icon.
2. `make digits MATCH=...` - templates of the new font (~20 min); it prints how many of each
   digit it found and a leave-one-out score, which should be (nearly) all right.
3. Cut a clip and `make run`, then compare `match.json` with what the broadcast shows.
4. The ball detector was fine-tuned on this broadcast. If deliveries are missed on the new one,
   label frames of it (`scripts/prepare_ball_labelling.py`, then `scripts/label_ball.py` in the
   browser, then `scripts/build_ball_finetune.py` and `scripts/finetune_ball.py`, ~1 h of GPU).

---

## 8. A commentator's voice

**Hear it before changing anything** - writes to `outputs/tts_test/` and plays it:

```bash
make say                                        # the configured voice, a sample line
make say TEXT="SIX! Into the stands!" ARGS="--style big"
make voices                                     # one sentence in 10 Kokoro voices, to pick by ear
make say ARGS="--voice hm_omega"                # try one of them (python scripts/say.py --list-voices: all 54)
make say ARGS='--engine f5 --ref-audio voices/me.m4a --ref-text "the exact words in the recording"'
```

To keep a Kokoro voice: `voice:` under `tts:` in `configs/commentary.yaml`, then
`python -m cricsense run --video data/<clip>.mp4 --voice --fresh voice render` and `make demo`.

### A cloned voice (F5-TTS)

Only a voice whose owner agreed - your own, or a friend's with their permission. Never a
celebrity's or any real person's voice without consent (it is a deepfake of them, and illegal in
many places).

1. Record 8-15 s of the person reading a sentence in a quiet room (phone or laptop is fine), one
   steady voice, no music. Convert it: `ffmpeg -i recording.m4a -ac 1 -ar 24000 voices/name.wav`
   (`voices/` is never committed - it is personal data).
2. In `configs/commentary.yaml`, under `tts:` set `engine: f5`, `ref_audio: voices/name.wav` and
   `ref_text: "..."` - exactly the words in the recording (a mismatch makes the voice slur).
3. `python -m cricsense run --video data/<clip>.mp4 --voice --fresh voice render`, then `make demo`.

Install once: `pip install -e ".[clone-voice]"` (it adds torchaudio 2.11 next to torch 2.14 - tested;
the environment before it is in `docs/history/ball-env-before-f5tts.txt` to roll back). The model
(~1.4 GB, non-commercial licence - fine for this project) downloads into `models/tts/f5` the first
time; on the GPU it speaks faster than real time. Back to the default voice: `engine: kokoro`.

## 9. Database, API and live mode

```bash
make db          # a local PostgreSQL for this project (no root)
python -m cricsense publish --video-id <clip>    # a clip's results into the database
make serve       # http://localhost:8000/docs - upload videos, read deliveries, scorecard, commentary
make live CLIP=<clip>    # plays the clip as if live: score, outcomes and radar as they happen
```

---

## When something goes wrong

| What you see | Why, and what to do |
|---|---|
| `calibrate` fails: "no frame shows the pitch's creases clearly enough" | The clip has too little of the main camera view (short, or mostly replays). Use `--cameras-from` (section 6) or a longer clip. |
| Calibration is wrong (green lines off the creases) | Delete `configs/calibration/<clip>.yaml` and `outputs/<clip>/`; calibrate with `--cameras-from`, or by hand: `python -m cricsense calibrate --video data/<clip>.mp4` and click the four crease corners. |
| A ball is missing from the fuse summary / "no clean score-bar update" | The score bar was hidden or misread around that ball. Check the frames in the broadcast; if the bar was on screen and clear, it is a reader problem (section 7 step 2 for the digits). |
| No length/line for a ball | The ball was not detected near the bowler's release or after the bounce (blur, a camera cut). Not fixable by settings; more labelled frames help (section 7 step 4). |
| Speed shows "≈ ... estimate" or nothing | No radar on the bar for that ball. The estimate is withheld when it would be faster than any ball ever bowled (a late-seen release). |
| `CUDA out of memory` | Another job holds the GPU. Run again later, or free the GPU. |
| Commentary is template-like | No API key, or the provider failed (the log says why); the fact-checker may also have rejected the model's lines - `commentary.json` gives each line's `fallback_reason`. |
| `make check` FAIL | Section 2. |

## Where things are

- `configs/` - every threshold and path. `golden.yaml` is the answer key; `clips.yaml` where each
  clip came from; `calibration/` one file per clip.
- `outputs/<clip>/` - everything a run produced. Safe to delete; it is regenerated.
- `docs/decisions.md` - what was measured and why each choice was made. Read the entry before
  changing a threshold: most have a measurement behind them.
- `docs/REPORT.md` - the technical report.
