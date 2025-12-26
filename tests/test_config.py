import pytest
from pathlib import Path
from src.config import (
    InferenceConfig,
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
)
from dataclasses import FrozenInstanceError
import src.config

## testy na nowym branchu


def test_base_dir_anchor():
    assert (BASE_DIR / "src").exists()


def test_dir_structure():
    assert "raw" in str(RAW_DATA_DIR)
    assert "processed" in str(PROCESSED_DATA_DIR)
    assert "models" in MODELS_DIR.parts
    assert "checkpoints" in MODELS_DIR.parts
    assert "logs" in str(LOGS_DIR)


def test_inference_config_paths():
    cfg = InferenceConfig()

    assert isinstance(cfg.model_path, Path)
    assert isinstance(cfg.logs_path, Path)
    assert isinstance(cfg.raw_data_path, Path)
    assert isinstance(cfg.processed_data_path, Path)


# INFERENCE
def test_inference_config_values():
    cfg = InferenceConfig(model_name="test_model")

    assert isinstance(cfg.model_name, str)
    assert cfg.model_name == "test_model"
    assert "test_model" in str(cfg.model_path)

    assert isinstance(cfg.image_size, tuple)
    assert len(cfg.image_size) == 2
    assert cfg.image_size == (250, 250)

    allowed_devices = ["cuda", "cpu", "mps"]
    assert isinstance(cfg.device, str)
    assert cfg.device in allowed_devices

    assert isinstance(cfg.batch_size, int)
    assert cfg.batch_size > 0
    assert (cfg.batch_size & cfg.batch_size - 1) == 0


def test_config_is_frozen():
    cfg = InferenceConfig()

    with pytest.raises(FrozenInstanceError):
        setattr(cfg, "model_name", "new_name")

    with pytest.raises(FrozenInstanceError):
        cfg.image_size = (200, 200)  # type: ignore
