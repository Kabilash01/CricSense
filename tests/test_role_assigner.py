"""Roles from pitch geometry (+ detector evidence), the replacement for infer_role_from_class."""

import numpy as np

from cricsense.identity.role_assigner import RoleAssigner, geometry_evidence
from cricsense.schema import Role

# A typical scene at the moment of delivery, in pitch metres.
SCENE = {
    1: ((1.0, 0.2), Role.STRIKER),
    2: ((-2.8, 0.4), Role.KEEPER),
    3: ((18.6, -1.7), Role.NON_STRIKER),
    4: ((19.2, 0.7), Role.BOWLER),
    5: ((22.0, 0.0), Role.UMPIRE),
    6: ((10.0, 25.0), Role.FIELDER),
    7: ((-5.0, -12.0), Role.FIELDER),
    8: ((0.5, 18.0), Role.FIELDER),
}


def run(frames: int, cls_for=lambda sid, f: None, conf=0.9, jitter=0.0, seed=0) -> dict[int, Role]:
    rng = np.random.default_rng(seed)
    ra, roles = RoleAssigner(), {}
    for f in range(frames):
        obs = {
            sid: ((x + rng.normal(0, jitter), y + rng.normal(0, jitter)), cls_for(sid, f), conf)
            for sid, ((x, y), _) in SCENE.items()
        }
        roles = ra.update(obs)
    return roles


def test_geometry_alone_assigns_every_role():
    assert run(30) == {sid: role for sid, (_, role) in SCENE.items()}


def test_positional_noise_is_tolerated():
    assert run(60, jitter=0.4) == {sid: role for sid, (_, role) in SCENE.items()}


def test_flickering_detector_class_does_not_flip_roles():
    # The legacy bug: the class flickers Batsman <-> Player every frame.
    flicker = lambda sid, f: ("Batsman" if f % 2 else "Player") if sid in (1, 3) else None
    roles = run(40, cls_for=flicker)
    assert roles[1] == Role.STRIKER and roles[3] == Role.NON_STRIKER


def test_unique_roles_are_unique():
    ra = RoleAssigner()
    # Two players both standing at the striker's crease: only one can be the striker.
    for _ in range(20):
        roles = ra.update({1: ((1.0, 0.1), None, 0.0), 2: ((1.3, -0.2), None, 0.0)})
    assert sorted(roles.values(), key=str).count(Role.STRIKER) == 1


def test_umpire_is_told_apart_from_non_striker_by_position():
    ev_ump = geometry_evidence(22.0, 0.0)
    ev_ns = geometry_evidence(18.6, 1.7)
    assert ev_ump[Role.UMPIRE] > ev_ump[Role.NON_STRIKER]
    assert ev_ns[Role.NON_STRIKER] > ev_ns[Role.UMPIRE]


def test_uncalibrated_frames_fall_back_to_detector_evidence():
    ra = RoleAssigner()
    for _ in range(30):
        roles = ra.update({1: (None, "Wicket_Keeper", 0.9), 2: (None, "Umpire", 0.9), 3: (None, "Player", 0.9)})
    assert roles == {1: Role.KEEPER, 2: Role.UMPIRE, 3: Role.FIELDER}


def test_moving_bowler_and_still_umpire_are_told_apart():
    # Both at the bowler's end, close together; only the bowler is running in.
    ra = RoleAssigner()
    rng = np.random.default_rng(1)
    for f in range(40):
        bowler = (26.0 - 0.2 * f + rng.normal(0, 0.15), 0.8 + rng.normal(0, 0.15))  # 5 m/s run-up
        umpire = (22.0 + rng.normal(0, 0.15), 0.0 + rng.normal(0, 0.15))  # still, noisy position
        roles = ra.update({1: (bowler, None, 0.0), 2: (umpire, None, 0.0)})
    assert roles == {1: Role.BOWLER, 2: Role.UMPIRE}
