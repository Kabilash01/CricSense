"""The stadium sound without the broadcast's own commentators, to put our commentary over.

Demucs (htdemucs, MIT licence, local) splits the sound into voices and the rest; the rest - the
crowd, bat on ball, the stumps - is kept. Its model (~80 MB) is models/demucs/955717e8-8726e21a.th
(`make models` downloads it from Meta's server; the name ends with the start of its sha256,
which demucs checks).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger("cricsense.render")

MODEL = "htdemucs"
SIG = "955717e8"  # htdemucs is this one model
PASSES = 2  # one pass left voice in 25 s of the 290 s between our lines (final_0051_6min); two, none


def decode(src: Path, offset_s: float, duration_s: float, sr: int) -> np.ndarray:
    """(n, 2) float32 samples of src from offset_s, via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{offset_s:.3f}", "-t", f"{duration_s:.3f}", "-i", str(src),
         "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.float32).reshape(-1, 2)


def without_voices(src: Path, offset_s: float, duration_s: float, out: Path, model_dir: Path, device: str) -> Path:
    """src's sound from offset_s for duration_s with the voices taken out -> out (wav). Reused
    while the source and the span are the same (out's .json beside it says which)."""
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    meta = {"source": str(src), "offset_s": round(offset_s, 3), "duration_s": round(duration_s, 3), "model": MODEL, "passes": PASSES}
    stamp = out.with_suffix(".json")
    if out.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == meta:
        return out
    if not any(model_dir.glob(f"{SIG}-*.th")):
        raise FileNotFoundError(f"no {MODEL} model in {model_dir}: make models")
    model = get_model(SIG, repo=model_dir).eval()
    rest = torch.from_numpy(decode(src, offset_s, duration_s, model.samplerate).T.copy())
    if not torch.cuda.is_available():
        device = "cpu"
    for _ in range(PASSES):  # the second pass takes out what the first left of the voices
        ref = rest.mean(0)  # normalised as demucs' own separate does
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            stems = apply_model(model, ((rest - mean) / std)[None], device=device, split=True, overlap=0.25, progress=False)[0]
        stems = stems * std + mean
        rest = sum(stems[i] for i, name in enumerate(model.sources) if name != "vocals")
    sf.write(out, np.clip(rest.T.numpy(), -1.0, 1.0), model.samplerate, subtype="PCM_16")
    stamp.write_text(json.dumps(meta))
    log.info("stadium sound without the broadcast's commentators: %s", out)
    return out
