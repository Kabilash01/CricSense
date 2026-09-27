import shutil

import pytest

from cricsense.config import ConfigError, find_root, load_config, video_id_from_path
from tests.conftest import REPO_ROOT


def test_loads_repo_config_with_relative_paths_resolved():
    cfg = load_config(REPO_ROOT)
    assert cfg.root == REPO_ROOT
    for p in vars(cfg.paths).values():
        assert p.is_absolute() and REPO_ROOT in p.parents
    assert cfg.output_dir_for("match_01") == REPO_ROOT / "outputs" / "match_01"
    assert cfg.calibration_path("match_01") == REPO_ROOT / "configs" / "calibration" / "match_01.yaml"
    assert cfg.video.resize is None
    assert "detector" in cfg.players and "detector" in cfg.ball


def test_find_root_walks_up_and_honours_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CRICSENSE_ROOT", raising=False)
    assert find_root(REPO_ROOT / "src" / "cricsense") == REPO_ROOT
    with pytest.raises(ConfigError, match="no configs/default.yaml"):
        find_root(tmp_path)
    monkeypatch.setenv("CRICSENSE_ROOT", str(REPO_ROOT))
    assert find_root(tmp_path) == REPO_ROOT


def _copy_configs(tmp_path):
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    return tmp_path / "configs"


def test_missing_config_file(tmp_path):
    (_copy_configs(tmp_path) / "ball.yaml").unlink()
    with pytest.raises(ConfigError, match="ball.yaml"):
        load_config(tmp_path)


def test_bad_resize(tmp_path):
    default = _copy_configs(tmp_path) / "default.yaml"
    default.write_text(default.read_text().replace("resize: null", "resize: [1280]"))
    with pytest.raises(ConfigError, match="video.resize"):
        load_config(tmp_path)


def test_missing_path_key(tmp_path):
    default = _copy_configs(tmp_path) / "default.yaml"
    default.write_text(default.read_text().replace("  ball_weights: models/ball_stump_best.pt\n", ""))
    with pytest.raises(ConfigError, match="paths.ball_weights"):
        load_config(tmp_path)


def test_video_id_is_file_stem():
    assert video_id_from_path("data/match_01.mp4") == "match_01"
