"""The release from the bowler's arm: the last top-of-swing before the ball is seen."""

from cricsense.pose.bowler import ArmSample, release_from_arm, swing_tops


def seq(pairs):
    return [ArmSample(f, h, (100.0, 50.0)) for f, h in pairs]


# Nortje, delivery 6 of the dev clip: a swing in the bound (5600), then the delivery (5607)
NORTJE = seq([(5600, 0.31), (5601, 0.28), (5602, -0.3), (5603, -0.31), (5604, -0.19), (5605, 0.0),
              (5606, 0.22), (5607, 0.33), (5608, -0.29), (5609, -0.38), (5610, -0.3), (5611, -0.1)])


def test_the_last_swing_is_the_delivery():
    assert [t.frame for t in swing_tops(NORTJE)] == [5600, 5607]
    assert release_from_arm(NORTJE)[0] == 5607.5
    assert release_from_arm(NORTJE, before=5604)[0] == 5600.5  # nothing later had been seen


def test_no_swing_no_release():
    assert release_from_arm(seq([(1, -0.1), (2, -0.15), (3, -0.12)])) is None
