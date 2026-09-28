import re
import subprocess
import sys

import pytest

from cricsense.cli import build_parser, main
from tests.conftest import REPO_ROOT

SUBCOMMANDS = ["calibrate", "scoreboard", "players", "ball", "fuse", "render", "live"]


def test_help_lists_every_subcommand():
    out = subprocess.run(
        [sys.executable, "-m", "cricsense", "--help"], capture_output=True, text=True, check=True
    ).stdout
    for name in SUBCOMMANDS:
        assert re.search(rf"^\s+{name}\s", out, re.MULTILINE), name


@pytest.mark.parametrize(
    "argv",
    [["ball", "--video", "x.mp4"], ["render", "--video-id", "x"], ["fuse", "--video-id", "x"]],
)
def test_unbuilt_commands_fail_loudly(argv, capsys):
    assert main(argv) == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_required_args_enforced():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["players"])


def test_no_hardcoded_absolute_paths_in_source():
    pattern = re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:\\[A-Za-z]|/home/|/Users/")
    offenders = [
        f"{p.relative_to(REPO_ROOT)}:{i}"
        for p in (REPO_ROOT / "src").rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []


def test_make_clip_command_is_frame_accurate(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import make_clip
    finally:
        sys.path.pop(0)
    assert make_clip.parse_time("1:02:03.5") == pytest.approx(3723.5)
    cmd = make_clip.build_command(tmp_path / "in.mp4", 3000.0, 60.0, tmp_path / "out.mp4")
    assert cmd[0] == "ffmpeg" and "copy" not in cmd  # re-encode, so the cut is not keyframe-snapped
