"""Match sheets (configs/matches/<match>.yaml): squads and keepers, the facts the video cannot show.

One sheet per match, so several matches live side by side. A video finds its match through
configs/clips.yaml - the broadcast it was cut from (`source`), or a `match:` given there - and a
whole-match video is its own match. A sheet may name its broadcaster's score-bar layout
(`scoreboard: <file in configs/>`, default scoreboard.yaml).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger("cricsense.match")
DEFAULT_LAYOUT = "scoreboard.yaml"


@dataclass(frozen=True)
class Team:
    code: str
    squad: tuple[str, ...]
    keeper: str | None
    name: str = ""  # "India"; the code when not given

    @property
    def display(self) -> str:
        return self.name or self.code


@dataclass(frozen=True)
class MatchSheet:
    teams: tuple[Team, ...]
    scoreboard: str = DEFAULT_LAYOUT  # the score-bar layout, a file in configs/

    @classmethod
    def from_yaml(cls, path: Path) -> MatchSheet:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        teams = tuple(
            Team(code=code, squad=tuple(t.get("squad") or ()), keeper=t.get("keeper"), name=t.get("name") or "")
            for code, t in (raw.get("teams") or {}).items()
        )
        if len(teams) != 2:
            raise ValueError(f"{path}: expected exactly two teams, got {[t.code for t in teams]}")
        for t in teams:
            if t.keeper is not None and t.keeper not in t.squad:
                raise ValueError(f"{path}: {t.code} keeper {t.keeper!r} is not in the squad")
        return cls(teams, raw.get("scoreboard") or DEFAULT_LAYOUT)

    @property
    def all_names(self) -> tuple[str, ...]:
        return tuple(n for t in self.teams for n in t.squad)

    def team_of(self, name: str | None) -> Team | None:
        for t in self.teams:
            if name in t.squad:
                return t
        return None

    def other(self, team: Team) -> Team:
        return next(t for t in self.teams if t.code != team.code)


def match_of(root: Path, video_id: str) -> str:
    """The match a video belongs to: a `match:` or the `source` broadcast it was cut from in
    configs/clips.yaml, else the video itself (a whole match)."""
    clips = root / "configs" / "clips.yaml"
    entry = ((yaml.safe_load(clips.read_text()) or {}).get("clips") or {}).get(video_id) if clips.is_file() else None
    if entry and entry.get("match"):
        return str(entry["match"])
    if entry and entry.get("source"):
        return Path(entry["source"]).stem
    return video_id


def sheet_path(root: Path, video_id: str) -> Path:
    return root / "configs" / "matches" / f"{match_of(root, video_id)}.yaml"


def load_sheet(root: Path, video_id: str) -> MatchSheet | None:
    """This video's match sheet; None (with a warning saying where it should be) if there is none."""
    path = sheet_path(root, video_id)
    if not path.is_file():
        log.warning("no match sheet for %s: expected %s - the two squads as the score bar prints them "
                    "(docs/RUNBOOK.md, a new match); without it no names, teams or score states", video_id, path)
        return None
    return MatchSheet.from_yaml(path)


def layout_path(root: Path, video_id: str) -> Path:
    """The score-bar layout for this video's broadcaster."""
    sheet = sheet_path(root, video_id)
    name = MatchSheet.from_yaml(sheet).scoreboard if sheet.is_file() else DEFAULT_LAYOUT
    return root / "configs" / name
