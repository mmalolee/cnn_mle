import pytest
from pathlib import Path
from src.cfg.training_config import TrainingConfig
from src.cfg.paths_config import PathsConfig
from dataclasses import FrozenInstanceError
from torchvision import transforms


# # --- FIXTURES -------------------------------------
@pytest.fixture(scope="function")
def common_params():
    params = {"model_name": "test_training_model", "device": "cuda"}

    return params


@pytest.fixture(scope="function")
def training_config_with_default(common_params):
    return TrainingConfig(**common_params, epochs=5, learning_rate=0.001, batch_size=16)


@pytest.fixture(scope="function")
def training_config_values(common_params):
    return TrainingConfig(
        **common_params, epochs=10, learning_rate=0.005, batch_size=32
    )


# # --- TESTS ----------------------------------------


@pytest.mark.parametrize(
    "attr, value",
    [
        ("rotation", 10),
        ("checkpoint_dir", PathsConfig().checkpoints_dir),
    ],
)
def test_default_values(training_config_with_default, attr, value):
    assert getattr(training_config_with_default, attr) == value


def test_checkpoint_path(training_config_values):
    assert training_config_values.checkpoint_dir.parts[-2:] == ("models", "checkpoints")


@pytest.mark.parametrize(
    "attr, value",
    [("epochs", 10), ("learning_rate", 0.005), ("batch_size", 32)],
)
def test_training_config_values(training_config_values, attr, value):
    assert getattr(training_config_values, attr) == value


@pytest.mark.parametrize(
    "attr, instance",
    [
        ("epochs", int),
        ("learning_rate", float),
        ("batch_size", int),
        ("rotation", int),
        ("checkpoint_dir", Path),
    ],
)
def test_is_instance(training_config_values, attr, instance):
    assert isinstance(getattr(training_config_values, attr), instance)


def test_is_frozen(training_config_values):
    with pytest.raises(FrozenInstanceError):
        setattr(training_config_values, "epochs", 15)


@pytest.mark.parametrize(
    "invalid_args",
    [{"epochs": 0}, {"learning_rate": -1}, {"learning_rate": 2}, {"batch_size": -1}],
)
def test_value_error(invalid_args, common_params):
    valid_args = {**common_params, "epochs": 1, "learning_rate": 0.01, "batch_size": 32}
    invalid_params = {**valid_args, **invalid_args}

    with pytest.raises(ValueError):
        TrainingConfig(**invalid_params)


def test_transformer(training_config_values):
    pil_steps = [type(t).__name__ for t in training_config_values.pil_transforms]
    tensor_steps = [type(t).__name__ for t in training_config_values.tensor_transforms]
    training_steps = [
        type(t).__name__ for t in training_config_values.training_transforms
    ]
    expected = [*pil_steps, *training_steps, *tensor_steps]
    training_transformer = [
        type(t).__name__ for t in training_config_values.training_transformer.transforms
    ]

    assert training_transformer == expected
