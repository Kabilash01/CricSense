"""`cricsense voice`: commentary.json -> commentary.wav (the spoken track) + voice.json.

The engine is configs/commentary.yaml tts.engine: kokoro (local, default) or gemini.

voice.json records where each line landed ({start_s, end_s, text, kind}) - later than
planned if the line before ran on - which the render uses for captions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cricsense.commentary.llm import LLMConfig, api_key
from cricsense.commentary.tts import ClipCache, TTSConfig, place, style_for, write_wav
from cricsense.config import Config
from cricsense.io import read_doc
from cricsense.schema import CommentaryDoc

log = logging.getLogger("cricsense.voice")


def run_voice(cfg: Config, video_id: str) -> tuple[Path, list[dict], list[str]]:
    out_dir = cfg.output_dir_for(video_id)
    doc = read_doc(out_dir / "commentary.json", CommentaryDoc)
    raw = (cfg.commentary.get("tts") or {})
    tcfg = TTSConfig(**{k: raw[k] for k in ("engine", "model", "voices", "voice") if k in raw})
    key = api_key(LLMConfig("gemini", tcfg.model, key_env=raw.get("key_env")), cfg.root) if tcfg.engine == "gemini" else None
    cache = ClipCache(out_dir / "tts")
    clips, texts, problems = [], [], []
    for line in doc.lines:
        style = style_for(line.kind, line.text)
        try:
            x = cache.get_or_make(tcfg, key, line.text, style, cfg.root)
        except Exception as e:  # one failed line must not cost the whole track
            problems.append(f"{line.t:6.1f}s  not spoken ({type(e).__name__}: {e})")
            continue
        clips.append((line.t, x))
        texts.append((line.text, line.kind, line.max_s))
    duration = doc.video.n_frames / doc.video.fps
    track, placed = place(clips, duration)
    wav = out_dir / "commentary.wav"
    write_wav(wav, track)
    lines = []
    for (start, end), (text, kind, max_s), (planned, _) in zip(placed, texts, clips):
        lines.append({"start_s": round(start, 2), "end_s": round(end, 2), "text": text, "kind": kind})
        if end - planned > max_s + 0.5:
            problems.append(f"{planned:6.1f}s  runs {end - planned - max_s:.1f}s past its slot")
    (out_dir / "voice.json").write_text(json.dumps({"voice": tcfg.voice, "model": tcfg.model, "lines": lines}, indent=1))
    return wav, lines, problems
