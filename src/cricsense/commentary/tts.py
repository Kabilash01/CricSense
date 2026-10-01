"""Text-to-speech for the commentary lines, and the commentary audio track.

Two engines:
  kokoro  local, free, no key: the Kokoro 82M model, int8 ONNX (~90 MB, ~0.4 GB of memory,
          CPU at ~0.5x real time); delivery is set by pace (it takes no style prompt);
  gemini  Gemini TTS over its REST Interactions endpoint (free tier, needs GEMINI_API_KEY);
          delivery is set by a style prompt.
Each line gets a delivery to match it - brisk as the ball is played, excited for a boundary
or a wicket, calm for the score - and is cached as a WAV file, so a re-run synthesises nothing. The track is the length of the clip with each line
placed at its time; a line that runs past the next one pushes it later, never overlaps it.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

RATE = 24000
GAP_S = 0.35

STYLES = {
    "ball": "a lively TV cricket commentator calling the ball as it happens, brisk and engaged",
    "big": "an excited TV cricket commentator, high energy, voice rising with the moment",
    "after": "a TV cricket commentator summing up, calm, warm and informative",
}


@dataclass(frozen=True)
class TTSConfig:
    engine: str = "kokoro"  # kokoro | gemini
    model: str = "models/tts/kokoro-v1.0.int8.onnx"  # kokoro: the ONNX file; gemini: the model id
    voices: str = "models/tts/voices-v1.0.bin"  # kokoro only
    voice: str = "bm_george"  # kokoro: a British male voice; gemini e.g. "Sadaltager"
    timeout_s: float = 60.0


PACE = {"ball": 1.1, "big": 1.2, "after": 1.0}  # kokoro: speaking rate per kind of line
_KOKORO = {}


def _kokoro(cfg: TTSConfig, root):
    key = (cfg.model, cfg.voices)
    if key not in _KOKORO:
        from kokoro_onnx import Kokoro

        _KOKORO[key] = Kokoro(str(root / cfg.model), str(root / cfg.voices))
    return _KOKORO[key]


def style_for(kind: str, text: str) -> str:
    if kind == "ball" and any(w in text for w in ("SIX", "FOUR", "OUT")):
        return STYLES["big"]
    return STYLES["after" if kind == "after" else "ball"]


def synthesize(cfg: TTSConfig, key: str | None, text: str, style: str, root=None) -> np.ndarray:
    """Mono float32 samples at RATE."""
    if cfg.engine == "kokoro":
        from pathlib import Path as _P

        kind = next((k for k, v in STYLES.items() if v == style), "ball")
        audio, sr = _kokoro(cfg, root or _P.cwd()).create(text, voice=cfg.voice, speed=PACE[kind], lang="en-gb")
        x = np.asarray(audio, np.float32)
        if sr != RATE:
            n = int(round(len(x) * RATE / sr))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        return x
    if key is None:
        raise RuntimeError("gemini TTS needs GEMINI_API_KEY")
    body = {
        "model": cfg.model,
        "input": [{"type": "user_input", "content": [{"type": "text", "text": text, "annotations": [{"type": "speech_metadata", "style": style, "speaker": "Commentator"}]}]}],
        "response_format": {"type": "audio", "mime_type": "audio/wav", "sample_rate": RATE},
        "generation_config": {"speech_config": {"speakers": [{"speaker": "Commentator", "voice": cfg.voice}]}},
        "stream": False,
    }
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        headers={"x-goog-api-key": key, "content-type": "application/json"},
        json=body,
        timeout=cfg.timeout_s,
    )
    r.raise_for_status()
    audio = [c for s in r.json().get("steps", []) if s.get("type") == "model_output" for c in s.get("content", []) if c.get("type") == "audio"]
    if not audio:
        raise RuntimeError("no audio in the TTS reply")
    return read_wav(base64.b64decode(audio[-1]["data"]))


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        if w.getframerate() != RATE:
            n = int(round(len(x) * RATE / w.getframerate()))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def write_wav(path: Path, x: np.ndarray, rate: int = RATE) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class ClipCache:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def path(self, cfg: TTSConfig, text: str, style: str) -> Path:
        h = hashlib.sha256(json.dumps([cfg.engine, cfg.model, cfg.voice, text, style]).encode()).hexdigest()[:20]
        return self.folder / f"{h}.wav"

    def get_or_make(self, cfg: TTSConfig, key: str | None, text: str, style: str, root=None) -> np.ndarray:
        p = self.path(cfg, text, style)
        if p.is_file():
            return read_wav(p.read_bytes())
        x = synthesize(cfg, key, text, style, root)
        write_wav(p, x)
        return x


def place(clips: list[tuple[float, np.ndarray]], duration_s: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Lay clips on a track at their times, pushing a clip later if the previous one runs on.
    Returns the track and (start, end) seconds of each clip as placed."""
    track = np.zeros(int(duration_s * RATE) + 1, np.float32)
    placed = []
    free_at = 0.0
    for t, x in clips:
        start = max(t, free_at)
        i = int(start * RATE)
        seg = x[: max(0, len(track) - i)]
        track[i : i + len(seg)] += seg
        end = start + len(x) / RATE
        placed.append((start, end))
        free_at = end + GAP_S
    peak = float(np.abs(track).max()) if len(track) else 0.0
    if peak > 0:
        track *= 0.89 / peak  # speech at a steady level; the mix sets the balance
    return track, placed
