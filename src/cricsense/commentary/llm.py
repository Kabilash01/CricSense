"""LLM commentary from the fact sheets, checked against them; the template is the fallback.

Provider-agnostic over plain HTTPS (no SDK): Anthropic, Gemini, or any OpenAI-compatible
endpoint (OpenAI, Groq, OpenRouter, a local server). The key comes from the environment or
the gitignored .env, never from a config file.

The model sees only the fact sheets. Every line it returns is checked before use:
  - every number in it must be a number in that ball's facts (no invented speeds or scores);
  - every player it names must be in that ball's facts (no one who was not involved);
  - it must fit the time before the next line (word budget).
A line that fails is replaced by the template line, and the reason is recorded.
Responses are cached by request, so a run can be repeated without new API calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests

WORDS_PER_S = 2.6  # a lively commentator; TTS runs at about this pace
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "fifty": 50, "hundred": 100, "century": 100}

SYSTEM = """You are a cricket TV commentator. You write short spoken lines for one ball at a time.
Rules - they are checked, and a line that breaks one is thrown away:
- Use ONLY the facts given for that ball. Do not invent anything: no shot names, fielders,
  field positions, conditions, history or statistics that are not in the facts.
- Every number you say must appear in that ball's facts. Say speeds exactly as given.
- If speed_source is "estimate", say "around" before the speed; if it is "radar", say it plainly.
- If bounce_confirmed is false, hedge the length ("looks full").
- If runs is null, the outcome is unknown: describe the delivery only.
- Name only the bowler, striker, non_striker or batting_team given for that ball.
- Say "SIX" / "FOUR" only if boundary says so; "out" only if wicket is true.
- A wicket's facts do not say HOW the batter was out: never say or imply bowled, caught, lbw,
  stumped, run out, an edge or a shot that got them out - only that the batter is out.
- footwork (front foot / back foot / in the crease) and swing (played / left) are measured; use them freely.
- shot/direction may be given (from video analysis); use them only if present, as given
  (e.g. "slog sweep", "leg side") - never name a shot or a fielding position that is not there.
- "over" is the ball just bowled, as overs.balls: "11.6" is the 6th ball of the 12th over. After
  the last ball of an over, overs_completed says how many overs are done; use that number.
- "after" is only for a boundary, a wicket, a milestone or the end of an over; otherwise null.
- Stay within max_words for each line. Plain text, no emojis, no hashtags.
Style: {style}
Return JSON only: {{"lines": [{{"delivery_id": <int>, "ball": "<line as the ball is played>", "after": "<score / over / milestone line, or null>"}}]}}"""


@dataclass(frozen=True)
class LLMConfig:
    provider: str  # anthropic | gemini | openai
    model: str
    style: str = "lively, warm, broadcast TV; varied phrasing"
    base_url: str | None = None  # openai-compatible endpoints
    key_env: str | None = None
    timeout_s: float = 60.0


DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}


def api_key(cfg: LLMConfig, root: Path) -> str | None:
    name = cfg.key_env or DEFAULT_KEY_ENV.get(cfg.provider, "")
    if os.environ.get(name):
        return os.environ[name]
    env = root / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


# ---- transport ---------------------------------------------------------------------------


def complete(cfg: LLMConfig, key: str, system: str, user: str, images: list[bytes] | None = None) -> str:
    """One model call; returns the text of the reply. images: JPEG bytes (for the VLM)."""
    images = images or []
    if cfg.provider == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(i).decode()}} for i in images]
        content.append({"type": "text", "text": user})
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": cfg.model, "max_tokens": 2000, "system": system, "messages": [{"role": "user", "content": content}]},
            timeout=cfg.timeout_s,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    if cfg.provider == "gemini":
        parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(i).decode()}} for i in images]
        parts.append({"text": user})
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.model}:generateContent",
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            json={"system_instruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": parts}],
                  "generationConfig": {"responseMimeType": "application/json"}},
            timeout=cfg.timeout_s,
        )
        r.raise_for_status()
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    if cfg.provider == "openai":
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(i).decode()}} for i in images]
        content.append({"type": "text", "text": user})
        r = requests.post(
            (cfg.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions",
            headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
            json={"model": cfg.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]},
            timeout=cfg.timeout_s,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise ValueError(f"unknown provider {cfg.provider!r}")


class Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, str] = json.loads(path.read_text()) if path.is_file() else {}

    def key(self, *parts: object) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, k: str) -> str | None:
        return self.data.get(k)

    def put(self, k: str, v: str) -> None:
        self.data[k] = v
        self.path.write_text(json.dumps(self.data, indent=1))


def parse_json(text: str) -> dict:
    """The reply's JSON object, tolerating a code fence or a sentence around it."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


# ---- the fact check ----------------------------------------------------------------------


def _numbers_in_facts(facts: dict) -> set[float]:
    nums: set[float] = set()
    if facts.get("over"):  # the over only as a whole ("11.6"), never its parts ("11 overs")
        nums.add(float(facts["over"]))
    for k, v in facts.items():
        if k == "over":
            continue
        for tok in re.findall(r"\d+(?:\.\d+)?", str(v)):
            x = float(tok)
            nums |= {x, float(round(x)), float(int(x))}
            if x >= 20:  # "around 155" for an estimate of 154
                nums.add(float(round(x / 5) * 5))
    return nums


# how a batter got out: never measured, so never said
_DISMISSAL_MODE = (
    r"\b(clean bowled|bowled (him|over)|bowls him|through the gate|stumps|timber|caught|catch|lbw|"
    r"leg before|stumped|run out|edges?|edged|nicks?|nicked|holes out|plumb|castled)\b"
)


def check_line(text: str, facts: dict, squad: set[str], max_words: int) -> str | None:
    """None if the line is faithful to the facts, else the reason it is not."""
    words = text.split()
    if len(words) > max_words:
        return f"too long ({len(words)} words, {max_words} fit)"
    allowed = _numbers_in_facts(facts)
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        if float(tok) not in allowed:
            return f"number {tok} is not in the facts"
    for w in re.findall(r"[A-Za-z]+", text.lower()):
        if w in NUMBER_WORDS and float(NUMBER_WORDS[w]) not in allowed and w not in ("one",):
            if not (w in ("four", "six") and facts.get("boundary") == w.upper()):
                return f"number '{w}' is not in the facts"
    named = {str(facts.get(k) or "").lower() for k in ("bowler", "striker", "non_striker", "batting_team")}
    lowered = f" {text.lower()} "
    for player in squad:
        p = player.lower()
        if re.search(rf"\b{re.escape(p)}\b", lowered) and p not in named:
            return f"names {player.title()}, who is not in the facts"
    if facts.get("boundary") is None and re.search(r"\b(six|four)\b!", lowered):
        return "calls a boundary the facts do not have"
    if not facts.get("wicket") and re.search(r"\b(wicket|bowled him|caught|lbw|stumped|run out)\b|\bout!|\b(is|he's|that's|and he's) out\b", lowered):
        return "calls a wicket the facts do not have"
    if facts.get("wicket") and re.search(_DISMISSAL_MODE, lowered):
        return "says how the batter was out; the facts do not"
    return None


def llm_lines(
    cfg: LLMConfig, key: str, facts: list[dict], budgets: dict[int, tuple[int, int]], cache: Cache
) -> dict[int, dict[str, str | None]]:
    """delivery_id -> {"ball": text, "after": text | None} from one model call for the clip."""
    payload = []
    for f in facts:
        b, a = budgets[f["delivery_id"]]
        payload.append({**{k: v for k, v in f.items() if not k.endswith("_frame")}, "max_words": {"ball": b, "after": a}})
    system = SYSTEM.format(style=cfg.style)
    user = "Facts (one entry per ball, in order)\n" + json.dumps(payload, indent=1)
    k = cache.key(cfg.provider, cfg.model, system, user)
    reply = cache.get(k)
    if reply is None:
        reply = complete(cfg, key, system, user)
        cache.put(k, reply)
    out = {}
    for item in parse_json(reply).get("lines", []):
        out[int(item["delivery_id"])] = {"ball": item.get("ball"), "after": item.get("after")}
    return out
