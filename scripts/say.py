"""Hear the commentary voice: speak a sentence with the configured TTS (or another), and play it.

    python scripts/say.py "Bumrah, a fuller one, 140.5 kph, and Hendricks is out!"
    python scripts/say.py "Four!" --style big --voice bm_lewis            # another Kokoro voice
    python scripts/say.py --audition "That is a beautiful cover drive."  # one sentence, many voices
    python scripts/say.py --list-voices
    python scripts/say.py "Four!" --engine f5 --ref-audio voices/me.m4a \\
        --ref-text "exactly the words said in the recording"           # a cloned voice (consent!)

Settings come from configs/commentary.yaml (tts:) unless given here. The audio is written to
outputs/tts_test/ and played (pass --no-play to only write it). A reference recording may be
.wav, .m4a, .mp3 or anything ffmpeg reads; it is converted to 24 kHz mono for F5.
--style: ball (brisk), big (a boundary or wicket: faster, excited), after (calm, the score).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cricsense.commentary.tts import STYLES, TTSConfig, synthesize, write_wav

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tts_test"
# commentator-like candidates: British and American male voices, and two Hindi voices speaking English
AUDITION = ("bm_george", "bm_lewis", "bm_daniel", "bm_fable", "am_michael", "am_adam", "am_onyx", "am_eric", "hm_omega", "hm_psi")


def play(path: Path) -> None:
    for player in (["pw-play"], ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(player[0]):
            subprocess.run([*player, str(path)], check=False)
            return
    print("  (no audio player found - open the file yourself)")


def as_wav(ref: Path) -> Path:
    """F5 wants a clean mono WAV; convert anything else with ffmpeg."""
    if ref.suffix.lower() == ".wav":
        return ref
    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / f"ref_{ref.stem}.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ref), "-ac", "1", "-ar", "24000", str(wav)], check=True)
    return wav


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="?", default="Bumrah runs in, a fuller one at 140 kph, and that is a beautiful cover drive for FOUR!")
    ap.add_argument("--engine", choices=("kokoro", "f5", "gemini"))
    ap.add_argument("--voice", help="kokoro voice name (see --list-voices)")
    ap.add_argument("--ref-audio", type=Path, help="f5: a recording of the voice to speak in")
    ap.add_argument("--ref-text", help="f5: exactly what is said in that recording")
    ap.add_argument("--style", choices=tuple(STYLES), default="ball")
    ap.add_argument("--audition", action="store_true", help="speak the text in several Kokoro voices, one after another")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--no-play", action="store_true")
    args = ap.parse_args(argv)

    raw = (yaml.safe_load((ROOT / "configs" / "commentary.yaml").read_text()) or {}).get("tts") or {}
    base = {k: raw[k] for k in ("engine", "model", "voices", "voice", "ref_audio", "ref_text") if k in raw}

    if args.list_voices:
        from kokoro_onnx import Kokoro

        kcfg = TTSConfig(**{k: v for k, v in base.items() if k in ("model", "voices")})
        names = Kokoro(str(ROOT / kcfg.model), str(ROOT / kcfg.voices)).get_voices()
        print("Kokoro voices (first letters: a = American, b = British, h = Hindi ...; f/m = female/male):")
        print("  " + "  ".join(names))
        return 0

    if args.voice or args.audition:
        base["engine"] = "kokoro"  # --voice names a Kokoro voice
    if args.voice:
        base["voice"] = args.voice
    if args.engine:
        base["engine"] = args.engine
    if args.ref_audio:
        base["ref_audio"] = str(as_wav(args.ref_audio.resolve()))  # absolute: the repo root is not prepended
    if args.ref_text:
        base["ref_text"] = args.ref_text

    key = None
    if base.get("engine") == "gemini":
        from cricsense.commentary.llm import LLMConfig, api_key

        key = api_key(LLMConfig("gemini", base.get("model", ""), key_env=raw.get("key_env")), ROOT)

    OUT.mkdir(parents=True, exist_ok=True)
    voices = AUDITION if args.audition else (base.get("voice"),)
    print(f'text: "{args.text}"  (style: {args.style})')
    for voice in voices:
        cfg = TTSConfig(**{**base, **({"voice": voice} if voice else {})})
        label = cfg.voice if cfg.engine == "kokoro" else (Path(cfg.ref_audio).stem if cfg.engine == "f5" else cfg.voice)
        t = time.perf_counter()
        x = synthesize(cfg, key, args.text, STYLES[args.style], ROOT)
        took = time.perf_counter() - t
        name = re.sub(r"[^a-z0-9]+", "_", f"{cfg.engine}_{label}_{args.style}".lower()).strip("_")
        path = OUT / f"{name}.wav"
        write_wav(path, x)
        print(f"  {cfg.engine:6s} {label:12s} {len(x) / 24000:4.1f} s of speech in {took:4.1f} s -> {path.relative_to(ROOT)}")
        if not args.no_play:
            play(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
