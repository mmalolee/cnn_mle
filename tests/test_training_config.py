import pytest
from pathlib import Path
from src.cfg.training_config import TrainingConfig
from src.cfg.paths_config import PathsConfig
from dataclasses import FrozenInstanceError


@pytest.fixture(scope="session")
def training_config():
    return TrainingConfig()


@pytest.mark.parametrize(
    "attr, default",
    [
        ("epochs", int),
        ("learning_rate", float),
        ("batch_size", int),
        ("checkopoint_dir", Path),
    ],
)
def test_is_instance(training_config, attr, default):
    assert isinstance(getattr(training_config, attr), default)


@pytest.mark.parametrize(
    "attr, default",
    [
        ("epochs", 10),
        ("learning_rate", 0.001),
        ("batch_size", 32),
        ("checkopoint_dir", PathsConfig().models_dir),
    ],
)
def test_values(training_config, attr, default):
    assert getattr(training_config, attr) == default


def test_is_frozen(training_config):
    with pytest.raises(FrozenInstanceError):
        setattr(training_config, "epochs", 15)


def test_checkpoint_path(training_config):
    assert training_config.checkopoint_dir.parts[-2:] == ("models", "checkpoints")
