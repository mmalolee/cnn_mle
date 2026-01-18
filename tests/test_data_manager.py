import pytest
from src.data_manager import LoaderData

from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataclasses import FrozenInstanceError


# # --- FIXTURES -------------------------------------
@pytest.fixture(scope="session")
def training_config():
    return TrainingConfig(
        model_name="test_for_data_loader",
        device="cuda",
        epochs=10,
        learning_rate=0.001,
        batch_size=32,
    )


# @pytest.fixture(scope="function")
# def get_fake_training_path(tmp_path):
#     fake_training_path = tmp_path / "Training"

#     fake_training_path.mkdir(parents=True, exist_ok=True)

#     (fake_training_path / "class_1").mkdir(parents=True)
#     (fake_training_path / "class_2").mkdir(parents=True)
#     return fake_training_path


# @pytest.fixture(scope="function")
# def get_fake_testing_path(tmp_path):
#     fake_testing_path = tmp_path / "Testing"

#     fake_testing_path.mkdir(parents=True, exist_ok=True)

#     (fake_testing_path / "class_1").mkdir(parents=True)
#     (fake_testing_path / "class_2").mkdir(parents=True)

#     return fake_testing_path


# @pytest.fixture(scope="session")
# def train_data_loader(
#     training_config, paths_config, get_fake_training_path, monkeypatch
# ):
#     training_data_loader = LoaderData(
#         path_config=paths_config, train_config=training_config
#     ).get_training_data_loader()

#     monkeypatch.setattr(training_data_loader.root_path, get_fake_training_path)


# @pytest.fixture(scope="session")
# def test_data_loader(
#     training_config, paths_config, get_fake_testing_path, monkeypatch
# ):
#     testing_data_loader = LoaderData(
#         path_config=paths_config, train_config=training_config
#     ).get_test_data_loader()

#     monkeypatch.setattr(testing_data_loader, get_fake_testing_path)


# # --- TESTS ----------------------------------------
