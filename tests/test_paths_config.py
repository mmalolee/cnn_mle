import pytest
from pathlib import Path
from src.cfg.paths_config import PathsConfig


# # --- FIXTURES -------------------------------------
@pytest.fixture
def temp_base(tmp_path):
    return tmp_path


@pytest.fixture(scope="function")
def paths(temp_base):
    return PathsConfig(temp_base)


# # --- TESTS ----------------------------------------
def test_base_dir(paths):
    assert isinstance(paths.base_dir, Path)
    assert paths.base_dir.exists()
    assert paths.base_dir.is_absolute()


@pytest.mark.parametrize(
    "attr_path",
    [
        "data_dir",
        "training_data_dir",
        "testing_data_dir",
        "models_dir",
        "checkpoints_dir",
        "logs_dir",
        "src_dir",
        "cfg_dir",
    ],
)
def test_is_instance_paths(paths, attr_path):
    path = getattr(paths, attr_path)
    assert isinstance(path, Path)


@pytest.mark.parametrize(
    "attr_path, expected",
    [
        ("data_dir", "data"),
        ("training_data_dir", "Training"),
        ("testing_data_dir", "Testing"),
        ("checkpoints_dir", "checkpoints"),
        ("models_dir", "models"),
        ("logs_dir", "logs"),
        ("src_dir", "src"),
        ("cfg_dir", "cfg"),
    ],
)
def test_paths_structure(paths, attr_path, expected):
    assert getattr(paths, attr_path).name == expected


@pytest.mark.parametrize(
    "attr_path",
    [
        "data_dir",
        "training_data_dir",
        "testing_data_dir",
        "models_dir",
        "logs_dir",
        "src_dir",
        "cfg_dir",
    ],
)
def test_is_absolute(paths, attr_path):
    path = getattr(paths, attr_path)
    assert path.is_absolute()


def test_checkpoints_path_structure(paths):
    assert paths.checkpoints_dir.parts[-2:] == ("models", "checkpoints")
    assert paths.checkpoints_dir.parent.parent == paths.base_dir


def test_models_path_structure(paths):
    assert paths.models_dir.name == "models"
    assert paths.models_dir.parent == paths.base_dir


def test_testing_data_path_structure(paths):
    assert paths.testing_data_dir.parent.parent == paths.base_dir


def test_training_data_path_structure(paths):
    assert paths.training_data_dir.parent.parent == paths.base_dir


def test_logs_path_structure(paths):
    assert paths.logs_dir.parent == paths.base_dir
