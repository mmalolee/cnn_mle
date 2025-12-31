import pytest
from src.cfg.model_config import ModelConfig
from dataclasses import FrozenInstanceError


@pytest.fixture(scope="session")
def model_config():
    return ModelConfig()


def test_is_instance(model_config):
    assert isinstance(model_config.img_size, int)
    assert isinstance(model_config.num_classes, int)
    assert isinstance(model_config.input_channels, int)


@pytest.mark.parametrize(
    "attr, default", [("img_size", 250), ("num_classes", 4), ("input_channels", 3)]
)
def test_values(model_config, attr, default):
    assert getattr(model_config, attr) == default


def test_is_frozen(model_config):
    with pytest.raises(FrozenInstanceError):
        setattr(model_config, "img_size", 5)
