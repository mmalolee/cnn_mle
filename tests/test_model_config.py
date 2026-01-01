import pytest
from src.cfg.model_config import ModelConfig
from dataclasses import FrozenInstanceError


@pytest.fixture(scope="function")
def model_config():
    return ModelConfig(img_size=250, num_classes=4, input_channels=3)


@pytest.mark.parametrize(
    "attr, value", [("img_size", 250), ("num_classes", 4), ("input_channels", 3)]
)
def test_values(model_config, attr, value):
    assert getattr(model_config, attr) == value


@pytest.mark.parametrize(
    "attr, default", [("img_size", int), ("num_classes", int), ("input_channels", int)]
)
def test_is_instance(model_config, attr, default):
    assert isinstance(getattr(model_config, attr), default)


def test_is_frozen(model_config):
    with pytest.raises(FrozenInstanceError):
        setattr(model_config, "img_size", 5)
