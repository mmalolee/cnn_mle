import pytest
from pathlib import Path
from dataclasses import FrozenInstanceError
import src.config as config

## testy na nowym branchu


def test_base_dir_exists():
    assert config.BASE_DIR.exists()


def test_data_paths_structure():
    assert config.DATA_DIR.parent == config.BASE_DIR
    assert config.RAW_DATA_DIR.parts[-2:] == ("data", "raw")
    assert config.RAW_DATA_DIR.parent == config.DATA_DIR
    assert config.PROCESSED_DATA_DIR.parts[-2:] == ("data", "processed")
    assert config.PROCESSED_DATA_DIR.parent == config.DATA_DIR


def test_models_paths_structure():
    assert config.MODELS_DIR.parts[-2:] == ("models", "checkpoints")
    assert config.MODELS_DIR.parent.parent == config.BASE_DIR


def test_logs_path_structure():
    assert config.LOGS_DIR.name == "logs"
    assert config.LOGS_DIR.parent == config.BASE_DIR


def test_inference_config_paths():
    cfg = config.InferenceConfig()

    assert isinstance(cfg.model_path, Path)
    assert isinstance(cfg.logs_path, Path)
    assert isinstance(cfg.raw_data_path, Path)
    assert isinstance(cfg.processed_data_path, Path)


# INFERENCE
def test_inference_config_values():
    cfg = config.InferenceConfig(model_name="test_model")

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
    cfg = config.InferenceConfig()

    with pytest.raises(FrozenInstanceError):
        setattr(cfg, "model_name", "new_name")

    with pytest.raises(FrozenInstanceError):
        cfg.image_size = (200, 200)  # type: ignore
