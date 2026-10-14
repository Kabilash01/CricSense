# CricSense: every task is one command. `make help` lists them; docs/RUNBOOK.md explains each.
# Variables can be set on the command line, e.g.  make run CLIP=st_bumrah_o2

PY    ?= $(if $(wildcard $(HOME)/miniconda3/envs/ball/bin/python),$(HOME)/miniconda3/envs/ball/bin/python,python)
CS    := $(PY) -m cricsense
MATCH ?= Cricket match/replay-final-sa-vs-ind.mp4
MATCH_ID := $(shell b="$(MATCH)"; b="$${b##*/}"; echo "$${b%.*}")
CLIP  ?=
START ?=
END   ?=
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

.PHONY: help doctor models test clip run run-novoice check check-quick save-golden benchmark sheet scan coverage demo digits serve live db

help:  ## list the commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

doctor:  ## is this machine ready? (programs, packages, models, GPU, keys)
	$(PY) scripts/doctor.py

models:  ## download the public models (YOLO, pose, Kokoro voice; ~300 MB); custom ones: README
	$(PY) -c "from ultralytics import YOLO; [YOLO(m) for m in ('models/yolov8m.pt', 'models/yolo11x-pose.pt')]"
	mkdir -p models/tts
	cd models/tts && for f in kokoro-v1.0.int8.onnx voices-v1.0.bin; do test -f $$f || curl -fL -o $$f https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$$f; done

test:  ## the unit tests (~30 s)
	$(PY) -m pytest -q

clip:  ## cut a clip:  make clip START=6840 END=7160 CLIP=my_clip   (seconds or HH:MM:SS)
	@test -n "$(CLIP)" -a -n "$(START)" -a -n "$(END)" || (echo "usage: make clip START=.. END=.. CLIP=name"; exit 2)
	$(PY) scripts/make_clip.py --video "$(MATCH)" --start $(START) --end $(END) --name $(CLIP)

run:  ## everything on one clip, with voice:  make run CLIP=my_clip
	@test -n "$(CLIP)" || (echo "usage: make run CLIP=name  (data/name.mp4)"; exit 2)
	$(CS) run --video data/$(CLIP).mp4 --voice

run-novoice:  ## everything on one clip, no speech
	@test -n "$(CLIP)" || (echo "usage: make run-novoice CLIP=name"; exit 2)
	$(CS) run --video data/$(CLIP).mp4

check:  ## regression check: re-run the verified clips, compare ball by ball (PASS/FAIL)
	$(PY) scripts/check.py

check-quick:  ## the same comparison on the outputs already there (seconds)
	$(PY) scripts/check.py --no-rerun

save-golden:  ## record current outputs as the expected answers - only after checking them by eye
	$(PY) scripts/check.py --save

benchmark:  ## 20 random clips of the match, no clicks; then the calibration sheet to check by eye
	$(PY) scripts/wide_eval.py --source "$(MATCH)" --legal 15 --cameras-from configs/calibration/final_0051_6min.yaml
	$(PY) scripts/calibration_sheet.py "outputs/we_*" --out outputs/calibration_sheet.jpg

sheet:  ## calibration sheet of any clips:  make sheet DIRS='outputs/st_*'
	$(PY) scripts/calibration_sheet.py $(or $(DIRS),"outputs/*") --out outputs/calibration_sheet.jpg

scan:  ## read the score bar over the whole match (~2-3 h)
	$(CS) scoreboard --video "$(MATCH)"

coverage:  ## how much of the whole match was read, and the final scores (after make scan)
	$(PY) scripts/coverage.py outputs/$(MATCH_ID)/scoreboard.json

demo:  ## the demo video from configs/demo.yaml (run each clip with voice first)
	$(PY) scripts/make_demo.py

digits:  ## templates of the score box's font (a new broadcaster; ~20 min)
	$(PY) scripts/harvest_digits.py --video "$(MATCH)" --step 2

db:  ## start the project's local PostgreSQL
	scripts/pg.sh start

serve:  ## the HTTP API on http://localhost:8000/docs (make db first)
	$(CS) serve --port 8000

live:  ## live mode on a clip:  make live CLIP=my_clip
	@test -n "$(CLIP)" || (echo "usage: make live CLIP=name"; exit 2)
	$(CS) live --video data/$(CLIP).mp4 --calibration configs/calibration/$(CLIP).yaml --out outputs/live/$(CLIP)_live.mp4
