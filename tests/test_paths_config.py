import pytest
from pathlib import Path
from src.paths_config import PathsConfig


@pytest.fixture(scope="session")
def paths():
    return PathsConfig()


def test_base_dir(paths):
    assert isinstance(paths.base_dir, Path)
    assert paths.base_dir.exists()
    assert paths.base_dir.is_absolute()


def test_is_instance_paths(paths):
    assert isinstance(paths.models_dir, Path)
    assert isinstance(paths.logs_dir, Path)
    assert isinstance(paths.raw_data_dir, Path)
    assert isinstance(paths.processed_data_dir, Path)


def test_data_paths_structure(paths):
    assert paths.data_dir.parent == paths.base_dir
    assert paths.raw_data_dir.parts[-2:] == ("data", "raw")
    assert paths.raw_data_dir.parent == paths.data_dir
    assert paths.processed_data_dir.parts[-2:] == ("data", "processed")
    assert paths.processed_data_dir.parent == paths.data_dir


def test_is_absolute(paths):
    assert paths.data_dir.is_absolute()
    assert paths.raw_data_dir.is_absolute()
    assert paths.processed_data_dir.is_absolute()


def test_models_paths_structure(paths):
    assert paths.models_dir.parts[-2:] == ("models", "checkpoints")
    assert paths.models_dir.parent.parent == paths.base_dir


def test_logs_path_structure(paths):
    assert paths.logs_dir.name == "logs"
    assert paths.logs_dir.parent == paths.base_dir
