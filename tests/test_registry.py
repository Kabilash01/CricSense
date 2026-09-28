"""Stable IDs. The headline legacy bug was 534 stable IDs for ~13 people."""

import numpy as np

from cricsense.identity.registry import Detection, PlayerRegistry, RegistryOptions

FPS = 25.0
N_PEOPLE = 13


def people_positions(frame: int, rng_seed: int = 0) -> np.ndarray:
    """13 people drifting slowly around a pitch (metres), deterministic."""
    rng = np.random.default_rng(rng_seed)
    start = rng.uniform([-10, -20], [30, 20], size=(N_PEOPLE, 2))
    heading = rng.uniform(-1, 1, size=(N_PEOPLE, 2))
    return start + heading * 1.2 * frame / FPS  # ~1.7 m/s at most


def box_at(px: float, py: float, pan: float = 0.0) -> tuple[float, float, float, float]:
    """A crude image box for a pitch position; `pan` shifts the whole image like a camera pan."""
    x, y = 960 + 25 * py + pan, 300 + 20 * px
    return (x - 15, y - 60, x + 15, y)


def churned_ids(frame: int, person: int, every: int) -> int:
    """The tracker hands this person a fresh ID every `every` frames (staggered)."""
    return person * 10_000 + (frame + 7 * person) // every


def test_tracker_id_churn_does_not_mint_new_players_when_calibrated():
    reg = PlayerRegistry(FPS)
    seen = set()
    for frame in range(0, 750):  # 30 s
        pos = people_positions(frame)
        pan = 300 * np.sin(frame / 60)  # the camera pans hard; pitch positions do not care
        dets = [
            Detection(churned_ids(frame, i, every=20), box_at(*pos[i], pan), tuple(pos[i]))
            for i in range(N_PEOPLE)
        ]
        seen |= set(reg.update(frame, dets).values())
    # 37 tracker IDs per person over 30 s; still one stable ID each.
    assert reg.n_players == N_PEOPLE and len(seen) == N_PEOPLE


def test_ids_survive_short_occlusions_in_image_space():
    reg = PlayerRegistry(FPS)
    for frame in range(0, 300):
        pos = people_positions(frame)
        visible = [i for i in range(N_PEOPLE) if not (i == 3 and 100 <= frame < 106)]  # person 3 hidden 6 frames
        dets = [Detection(churned_ids(frame, i, every=40), box_at(*pos[i])) for i in visible]
        reg.update(frame, dets)
    assert reg.n_players == N_PEOPLE


def test_class_flips_cannot_split_a_player():
    # The detector class is not an input at all: this is the structural fix for B2.
    reg = PlayerRegistry(FPS)
    sids = {reg.update(f, [Detection(1 if f < 5 else 2, box_at(0, 0), (0.0, 0.0))])[1 if f < 5 else 2] for f in range(10)}
    assert sids == {1}


def test_crossing_players_keep_their_ids():
    reg = PlayerRegistry(FPS)
    ids = []
    for frame in range(40):
        a = (5.0 + 0.1 * frame, 0.0)  # walking toward each other and past
        b = (9.0 - 0.1 * frame, 0.5)
        raw = frame // 5  # tracker reissues both IDs every 5 frames
        sid = reg.update(frame, [Detection(100 + raw, box_at(*a), a), Detection(200 + raw, box_at(*b), b)])
        ids.append((sid[100 + raw], sid[200 + raw]))
    assert set(ids) == {(1, 2)}


def test_far_away_newcomer_is_a_new_player():
    reg = PlayerRegistry(FPS)
    reg.update(0, [Detection(1, box_at(0, 0), (0.0, 0.0))])
    assert reg.update(1, [Detection(2, box_at(15, 10), (15.0, 10.0))]) == {2: 2}


def test_no_image_space_links_across_a_camera_cut():
    reg = PlayerRegistry(FPS)
    reg.update(0, [Detection(1, box_at(0, 0))])  # uncalibrated frame
    reg.camera_cut()
    # Same pixels after a cut mean nothing: a different shot, so a different person.
    assert reg.update(1, [Detection(1, box_at(0, 0))]) == {1: 2}


def test_pitch_links_do_survive_a_cut():
    reg = PlayerRegistry(FPS)
    reg.update(0, [Detection(1, box_at(10, 1), (10.0, 1.0))])
    reg.camera_cut()
    assert reg.update(20, [Detection(9, box_at(10.4, 1, pan=500), (10.4, 1.0))]) == {9: 1}


def test_lost_players_expire():
    reg = PlayerRegistry(FPS, RegistryOptions(max_lost_frames=10))
    reg.update(0, [Detection(1, box_at(0, 0), (0.0, 0.0))])
    assert reg.update(30, [Detection(2, box_at(0, 0), (0.0, 0.0))]) == {2: 2}
