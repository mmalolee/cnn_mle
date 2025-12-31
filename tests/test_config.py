import pytest
from pathlib import Path
from dataclasses import FrozenInstanceError
from src.config import InferenceConfig, TrainingConfig


# # INFERENCE
# def test_inference_config_model_name():
#     cfg = config.InferenceConfig(model_name="test_model")

#     assert isinstance(cfg.model_name, str)
#     assert cfg.model_name == "test_model"
#     assert "test_model" in cfg.model_path.name


# def test_inference_config_image_size():
#     cfg = config.InferenceConfig()

#     assert isinstance(cfg.image_size, tuple)
#     assert len(cfg.image_size) == 2
#     assert cfg.image_size == (250, 250)


# def test_inference_config_allowed_devices():
#     available_devices = config.InferenceConfig().ALLOWED_DEVICES
#     expected = ("cpu", "cuda", "mps")

#     assert isinstance(available_devices, tuple)
#     assert expected == available_devices

#     cfg = config.InferenceConfig()

#     assert cfg.device in expected


# def test_config_raises_error_on_invalid_device():
#     with pytest.raises(ValueError):
#         config.InferenceConfig(device="pegasus")


# def test_inference_config_batch_size():
#     cfg = config.InferenceConfig()

#     assert isinstance(cfg.batch_size, int)
#     assert cfg.batch_size > 0
#     assert (cfg.batch_size & cfg.batch_size - 1) == 0


# def test_config_is_frozen():
#     cfg = config.InferenceConfig()

#     with pytest.raises(FrozenInstanceError):
#         setattr(cfg, "model_name", "new_name")

#     with pytest.raises(FrozenInstanceError):
#         cfg.image_size = (200, 200)  # type: ignore
