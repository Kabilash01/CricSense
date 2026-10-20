"""`cricsense commentary`: spoken-style lines per ball -> commentary.json.

Facts from ball.json, events.json and match.json; optionally shot type and direction from a
VLM; lines from an LLM checked against the facts, or from templates. Each line has a start
time and the time available before the next one, which the TTS and the mixer respect.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cricsense.commentary.facts import BallFacts, build_facts
from cricsense.commentary.llm import WORDS_PER_S, Cache, LLMConfig, api_key, check_line, llm_lines
from cricsense.commentary.templates import after_line, ball_line
from cricsense.commentary.vlm import contact_frames, name_shot, read_shot
from cricsense.config import Config
from cricsense.identity.match_sheet import load_sheet
from cricsense.io import read_doc, write_doc
from cricsense.schema import BallDoc, CommentaryDoc, CommentaryLine, EventsDoc, MatchDoc, PlayersDoc, PoseDoc, Role

log = logging.getLogger("cricsense.commentary")

GAP_S = 0.4  # between two lines


def _llm_cfg(raw: dict) -> LLMConfig:
    return LLMConfig(
        provider=raw.get("provider", "anthropic"),
        model=raw["model"],
        style=raw.get("style", LLMConfig.style),
        base_url=raw.get("base_url"),
        key_env=raw.get("key_env"),
    )


def _duration_s(text: str) -> float:
    return len(text.split()) / WORDS_PER_S + 0.3


def schedule(f: BallFacts, fps: float, n_frames: int) -> tuple[int, int, int]:
    """(ball line start, after line earliest start, end of this ball's window) in frames."""
    start = (f.contact_frame + 3) if f.contact_frame is not None else f.release_frame + int(0.9 * fps)
    end = (f.next_release_frame - int(1.0 * fps)) if f.next_release_frame is not None else n_frames - 1
    after = f.outcome_frame if f.outcome_frame is not None else end
    return min(start, end), min(max(after, start), end), max(end, start + 1)


def run_commentary(cfg: Config, video_id: str, engine: str | None = None, vlm: bool | None = None) -> tuple[CommentaryDoc, Path]:
    out_dir = cfg.output_dir_for(video_id)
    ball = read_doc(out_dir / "ball.json", BallDoc)
    events = read_doc(out_dir / "events.json", EventsDoc)
    match = read_doc(out_dir / "match.json", MatchDoc) if (out_dir / "match.json").is_file() else None
    sheet = load_sheet(cfg.root, video_id)
    squad = set(sheet.all_names) if sheet else set()
    fps, n_frames = ball.video.fps, ball.video.n_frames
    pose = read_doc(out_dir / "pose.json", PoseDoc) if (out_dir / "pose.json").is_file() else None
    facts = build_facts(ball, events, match, sheet, pose)

    ccfg = cfg.commentary
    engine = engine or ccfg.get("engine", "auto")
    llm_cfg = _llm_cfg(ccfg["llm"]) if ccfg.get("llm") else None
    key = api_key(llm_cfg, cfg.root) if llm_cfg else None
    if engine == "llm" and key is None:
        raise RuntimeError(f"engine llm: no API key ({llm_cfg.key_env or llm_cfg.provider.upper() + '_API_KEY'}) in the environment or .env")
    use_llm = engine == "llm" or (engine == "auto" and key is not None)
    cache = Cache(out_dir / "llm_cache.json")

    vcfg = ccfg.get("vlm") or {}
    if (vlm if vlm is not None else vcfg.get("enabled", False)) and use_llm:
        vlm_cfg = _llm_cfg(vcfg)
        vkey = api_key(vlm_cfg, cfg.root)
        video = next(cfg.paths.video_dir.glob(f"{video_id}.*"))
        striker_boxes: dict[int, tuple] = {}
        if (out_dir / "players.json").is_file():
            for pf in read_doc(out_dir / "players.json", PlayersDoc).frames:
                if pf.role == Role.STRIKER:
                    striker_boxes[pf.frame_id] = pf.box

        def striker_box(frame: int):
            return striker_boxes.get(frame)

        for f in facts:
            if vkey and (f.bat_contact or (f.off_bat or 0) > 0):
                centre = f.contact_frame if f.contact_frame is not None else f.release_frame + int(0.8 * fps)
                try:
                    shot = read_shot(vlm_cfg, vkey, contact_frames(video, centre, striker_box), cache)
                except Exception as e:  # the VLM is an extra; commentary goes on without it
                    log.warning("VLM failed for delivery %d: %s", f.delivery_id, e)
                    shot = None
                if shot and f.wicket:
                    # nothing measured says how the batter got out; a shot beside "out" reads as the
                    # mode ("Hendricks flicks it leg side, out!" - he was bowled)
                    shot = None
                if shot and f.swing == "left":
                    shot = None  # the measured swing says the ball was left alone
                if shot and f.footwork in ("front foot", "back foot"):
                    # the measured footwork outranks what the VLM thought it saw
                    renamed = name_shot({**shot["observed"], "foot": f.footwork.split()[0]})
                    shot = {**shot, "shot": renamed} if renamed else None
                if shot:
                    f.shot, f.direction, f.shot_source = shot["shot"], shot["direction"], "vlm"

    caps = ccfg.get("max_words", {"ball": 22, "after": 24})
    plan = {f.delivery_id: schedule(f, fps, n_frames) for f in facts}
    budgets = {}
    for f in facts:
        s, a, e = plan[f.delivery_id]
        budgets[f.delivery_id] = (
            max(6, min(caps["ball"], int(((a if f.outcome_frame else e) - s) / fps * WORDS_PER_S))),
            max(6, min(caps["after"], int((e - a) / fps * WORDS_PER_S))),
        )

    generated: dict[int, dict] = {}
    engine_name = "template"
    if use_llm:
        try:
            generated = llm_lines(llm_cfg, key, [f.as_dict() for f in facts], budgets, cache)
            engine_name = f"{llm_cfg.provider}:{llm_cfg.model}"
        except Exception as e:
            log.warning("LLM failed, using templates for every line: %s", e)
            engine_name = f"template (LLM failed: {type(e).__name__})"

    lines: list[CommentaryLine] = []
    for f in facts:
        start, after_at, end = plan[f.delivery_id]
        fd = f.as_dict()
        for kind, template, budget in (("ball", ball_line(f), budgets[f.delivery_id][0]), ("after", after_line(f), budgets[f.delivery_id][1])):
            text, source, reason = template, "template", None
            candidate = (generated.get(f.delivery_id) or {}).get(kind)
            if kind == "after" and template is None:
                candidate = None  # nothing worth an extra line: no boundary, wicket, milestone or over end
            if generated:
                if candidate:
                    reason = check_line(candidate, fd, squad, budget)
                    if reason is None:
                        text, source = candidate.strip(), "llm"
                elif template:
                    reason = "no line from the LLM"
            if not text:
                continue
            if kind == "ball":
                at = start
            else:
                prev_end = lines[-1].frame_id + int((_duration_s(lines[-1].text) + GAP_S) * fps) if lines and lines[-1].delivery_id == f.delivery_id else start
                at = max(after_at, prev_end)
                if at >= end:
                    continue  # no time left before the next ball
            limit = (after_at if kind == "ball" and f.outcome_frame is not None else end) - at
            lines.append(
                CommentaryLine(
                    delivery_id=f.delivery_id,
                    frame_id=at,
                    t=at / fps,
                    kind=kind,
                    text=text,
                    source=source,
                    fallback_reason=reason if source == "template" else None,
                    max_s=max(limit, int(fps)) / fps,
                )
            )
    doc = CommentaryDoc(video=ball.video, engine=engine_name, lines=lines, facts=[f.as_dict() for f in facts])
    path = write_doc(doc, out_dir / "commentary.json")
    return doc, path


def summarize(doc: CommentaryDoc, path: Path) -> list[str]:
    out = [f"engine: {doc.engine}"]
    for line in doc.lines:
        mm, ss = divmod(line.t, 60)
        tag = "" if line.source == "llm" else ("  [template: " + line.fallback_reason + "]" if line.fallback_reason else "")
        out.append(f"{int(mm):02d}:{ss:04.1f}  {line.text}{tag}")
    shots = [f for f in doc.facts if f.get("shot")]
    if shots:
        out.append("shots (posture read by the VLM, named by rule): " + ", ".join(f"{f['over']} {f['shot']} ({f['direction']})" for f in shots))
    out.append(f"commentary: {path}")
    return out
