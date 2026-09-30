"""Match sheet (configs/match.yaml): squads and keepers, the facts the video cannot show."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
        return cls(teams)

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
