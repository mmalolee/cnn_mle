import pytest
from src.cfg.model_config import ModelConfig
from dataclasses import FrozenInstanceError


# # --- FIXTURES -------------------------------------
@pytest.fixture(scope="function")
def model_config():
    return ModelConfig()


# # --- TESTS ----------------------------------------
@pytest.mark.parametrize(
    "attr, default",
    [("model_name", "CNN"), ("num_classes", 4), ("input_channels", 3)],
)
def test_values(model_config, attr, default):
    assert getattr(model_config, attr) == default


@pytest.mark.parametrize(
    "attr, default",
    [("model_name", str), ("num_classes", int), ("input_channels", int)],
)
def test_is_instance(model_config, attr, default):
    assert isinstance(getattr(model_config, attr), default)


def test_is_frozen(model_config):
    with pytest.raises(FrozenInstanceError):
        setattr(model_config, "img_size", 5)
