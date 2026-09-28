from cricsense.identity.vote_tracker import JerseyVoteTracker


def test_two_digit_number_locks_after_agreeing_votes():
    v = JerseyVoteTracker()
    assert v.add(1, [("23", 0.9)]) is None
    assert v.add(1, [("23", 0.9)]) is None
    lock = v.add(1, [("23", 0.9), ("28", 0.4)])
    assert lock is not None and lock.number == "23" and v.locked(1) == lock


def test_single_digit_number_can_lock():
    # Legacy: 50 votes for "7" never locked.
    v = JerseyVoteTracker()
    locks = [v.add(1, [("7", 1.0)]) for _ in range(6)]
    assert any(locks) and v.locked(1).number == "7"


def test_single_digits_need_more_evidence_than_two_digits():
    v = JerseyVoteTracker()
    for _ in range(3):
        v.add(1, [("7", 1.0)])
        v.add(2, [("17", 1.0)])
    assert v.locked(1) is None and v.locked(2) is not None


def test_disagreement_prevents_a_lock():
    v = JerseyVoteTracker()
    for n in ["23", "28", "23", "28", "23", "28"]:
        v.add(1, [(n, 0.9)])
    assert v.locked(1) is None


def test_junk_and_low_confidence_reads_are_ignored():
    v = JerseyVoteTracker()
    for cands in ([], [("0", 0.9)], [("123", 0.9)], [("4a", 0.9)], [("45", 0.1)]):
        assert v.add(1, cands) is None
    assert v.locked(1) is None


def test_ocr_candidate_lists_are_accepted():
    # Legacy crashed: AttributeError: 'list' object has no attribute 'isdigit'
    v = JerseyVoteTracker()
    v.add(1, [("45", 0.8), ("4", 0.5)])  # does not raise
