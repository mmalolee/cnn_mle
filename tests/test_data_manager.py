import pytest
from src.data_manager import LoaderData
from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig
from PIL import Image
import torch
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


@pytest.fixture(scope="function")
def paths_config(tmp_path):
    return PathsConfig(base_dir=tmp_path)


@pytest.fixture(scope="function")
def loader_data(paths_config, training_config):
    for cls in ["class_1", "class_2"]:
        training_path = paths_config.training_data_dir / cls
        testing_path = paths_config.testing_data_dir / cls

        training_path.mkdir(parents=True, exist_ok=True)
        testing_path.mkdir(parents=True, exist_ok=True)

        fake_img = Image.new("RGB", (250, 250), color="black")

        fake_img.save(training_path / "dummy.jpg")
        fake_img.save(testing_path / "dummy.jpg")

    return LoaderData(path_config=paths_config, train_config=training_config)


# # --- TESTS ----------------------------------------
def test_training_data_loader_instance(loader_data):
    training_loader = loader_data.get_training_data_loader()

    assert isinstance(training_loader, DataLoader)


def test_testing_data_loader_instance(loader_data):
    testing_loader = loader_data.get_test_data_loader()

    assert isinstance(testing_loader, DataLoader)


def test_training_data_loader_img(loader_data):
    training_loader = loader_data.get_training_data_loader()

    assert len(training_loader.dataset) == 2


def test_testing_data_loader_img(loader_data):
    testing_loader = loader_data.get_test_data_loader()

    assert len(testing_loader.dataset) == 2


def test_loaders_have_different_transformers(loader_data):
    train_loader = loader_data.get_training_data_loader()
    test_loader = loader_data.get_test_data_loader()

    assert train_loader != test_loader


def test_training_data_loader_error(paths_config, training_config):
    with pytest.raises(FileNotFoundError):
        LoaderData(
            path_config=paths_config, train_config=training_config
        ).get_training_data_loader()


def test_testing_data_loader_error(paths_config, training_config):
    with pytest.raises(FileNotFoundError):
        LoaderData(
            path_config=paths_config, train_config=training_config
        ).get_test_data_loader()


def test_training_item_is_tensor(loader_data):
    training_loader = loader_data.get_training_data_loader()
    img = training_loader.dataset[0][0]

    assert isinstance(img, torch.Tensor)


def test_testing_item_is_tensor(loader_data):
    test_loader = loader_data.get_test_data_loader()
    img = test_loader.dataset[0][0]

    assert isinstance(img, torch.Tensor)


def test_training_item_is_int(loader_data):
    training_loader = loader_data.get_training_data_loader()
    label = training_loader.dataset[0][1]

    assert isinstance(label, int)


def test_testing_item_is_int(loader_data):
    test_loader = loader_data.get_test_data_loader()
    label = test_loader.dataset[0][1]

    assert isinstance(label, int)


def test_training_class_mapping(loader_data):
    training_loader = loader_data.get_training_data_loader()
    training_dataset = training_loader.dataset
    expected_classes = ["class_1", "class_2"]

    assert all(cls in training_dataset.classes for cls in expected_classes)


def test_training_index_mapping(loader_data):
    training_loader = loader_data.get_training_data_loader()
    training_dataset = training_loader.dataset

    assert training_dataset.class_to_idx["class_1"] == 0
    assert training_dataset.class_to_idx["class_2"] == 1


def test_testing_class_mapping(loader_data):
    testing_loader = loader_data.get_test_data_loader()
    testing_dataset = testing_loader.dataset
    expected_classes = ["class_1", "class_2"]

    assert all(cls in testing_dataset.classes for cls in expected_classes)


def test_testing_index_mapping(loader_data):
    testing_loader = loader_data.get_test_data_loader()
    testing_dataset = testing_loader.dataset

    assert testing_dataset.class_to_idx["class_1"] == 0
    assert testing_dataset.class_to_idx["class_2"] == 1
