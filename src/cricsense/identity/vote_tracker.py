"""Temporal voting on jersey-number reads, locking a number once the votes agree.

Migrated from CricSense-AI src/jersey/jersey_vote_tracker.py with two fixes:
  - single-digit numbers (1-9) could never lock: a vote was only accepted once that number
    already had 5 votes, which it could never collect;
  - it expected a bare string, but the OCR returns (digits, confidence) candidates.
Single digits are still held to a stricter bar, because a lone digit is often half of a
partly hidden two-digit number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class JerseyLock:
    number: str
    votes: int
    agreement: float  # share of weighted votes for the locked number


@dataclass
class JerseyVoteTracker:
    min_votes: int = 3  # agreeing reads, counted
    min_agreement: float = 0.66  # confidence-weighted share of all reads; 3-vs-2 is not enough
    min_votes_single_digit: int = 5
    min_agreement_single_digit: float = 0.7
    min_confidence: float = 0.3  # OCR candidates below this are ignored

    def __post_init__(self) -> None:
        self._counts: dict[int, Counter] = {}
        self._weights: dict[int, Counter] = {}
        self._locked: dict[int, JerseyLock] = {}

    def locked(self, sid: int) -> JerseyLock | None:
        return self._locked.get(sid)

    def add(self, sid: int, candidates: list[tuple[str, float]]) -> JerseyLock | None:
        """Add one crop's OCR candidates. Returns the lock if this vote caused it, else None."""
        if sid in self._locked:
            return None
        valid = [(d, c) for d, c in candidates if d.isdigit() and 1 <= int(d) <= 99 and c >= self.min_confidence]
        if not valid:
            return None
        number, conf = max(valid, key=lambda dc: dc[1])
        number = str(int(number))
        counts = self._counts.setdefault(sid, Counter())
        weights = self._weights.setdefault(sid, Counter())
        counts[number] += 1
        weights[number] += conf
        return self._try_lock(sid, counts, weights)

    def _try_lock(self, sid: int, counts: Counter, weights: Counter) -> JerseyLock | None:
        number, weight = weights.most_common(1)[0]
        agreement = weight / sum(weights.values())
        single = len(number) == 1
        need_votes = self.min_votes_single_digit if single else self.min_votes
        need_agreement = self.min_agreement_single_digit if single else self.min_agreement
        if counts[number] >= need_votes and agreement >= need_agreement:
            lock = JerseyLock(number=number, votes=counts[number], agreement=round(agreement, 2))
            self._locked[sid] = lock
            return lock
        return None
