import pytest
from src.cfg.inference_config import InferenceConfig
from dataclasses import FrozenInstanceError
from torchvision import transforms


# # --- FIXTURES -------------------------------------
@pytest.fixture(scope="function")
def inference_config():
    return InferenceConfig(model_name="test_model", device="cuda")


# # --- TESTS ----------------------------------------
@pytest.mark.parametrize(
    "attr, value",
    [("model_name", "test_model"), ("device", "cuda")],
)
def test_values(inference_config, attr, value):
    assert getattr(inference_config, attr) == value


@pytest.mark.parametrize(
    "attr, instance",
    [("model_name", str), ("ALLOWED_DEVICES", tuple), ("device", str)],
)
def test_is_instance(inference_config, attr, instance):
    assert isinstance(getattr(inference_config, attr), instance)


def test_is_allowed_device(inference_config):
    assert inference_config.device in inference_config.ALLOWED_DEVICES


def test_allowed_devices_length(inference_config):
    assert len(inference_config.ALLOWED_DEVICES) == 3


def test_transformer_instance(inference_config):
    assert isinstance(inference_config.transformer, transforms.Compose)


def test_is_frozen(inference_config):
    with pytest.raises(FrozenInstanceError):
        setattr(inference_config, "device", "pegasus")


def test_invalid_device_error():
    with pytest.raises(ValueError):
        InferenceConfig(model_name="test", device="pegasus")


def test_transformer_instances(inference_config):
    types = [type(t) for t in inference_config.transformer.transforms]
    assert transforms.Resize in types
    assert transforms.ToTensor in types
