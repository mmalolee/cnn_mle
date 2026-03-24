import pytest
import torch

from src.architectures.cnn import CNN
from src.cfg.model_config import ModelConfig


# # --- FIXTURES -------------------------------------
@pytest.fixture(scope="session")
def get_model():
    return CNN(model_cfg=ModelConfig())


# # --- TESTS ----------------------------------------
def test_features_len(get_model):
    assert len(get_model.features) == 3


def test_classification_len(get_model):
    assert len(get_model.classification) == 3


def test_batch_consistency(get_model):
    x = torch.randn(16, 3, 250, 250)
    output = get_model(x)

    assert output.shape[0] == 16


def test_class_consistency(get_model):
    x = torch.randn(16, 3, 250, 250)
    output = get_model(x)

    assert output.shape[1] == get_model.model_cfg.num_classes
