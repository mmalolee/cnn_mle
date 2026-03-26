import os
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoaderData:
    path_config: PathsConfig
    train_config: TrainingConfig

    def _create_loader(
        self, root_path: Path, transformer: transforms.Compose, shuffle: bool
    ) -> DataLoader:
        if not root_path.exists():
            logger.critical(f"Data path does not exist: {root_path}")
            raise FileNotFoundError(f"Data path does not exist: {root_path}.")

        dataset = datasets.ImageFolder(root=root_path, transform=transformer)

        if len(dataset) == 0:
            msg = f"No images found in {root_path}."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            f"Loaded {len(dataset)} images from {root_path}. Classes found: {dataset.classes}"
        )

        cpus = os.cpu_count()
        workers = (cpus // 2) if cpus is not None else 1

        loader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
        )

        return loader

    def get_training_data_loader(self) -> DataLoader:
        logger.info("Initializing training data loader...")
        return self._create_loader(
            root_path=self.path_config.training_data_dir,
            transformer=self.train_config.training_transformer,
            shuffle=True,
        )

    def get_test_data_loader(self) -> DataLoader:
        logger.info("Initializing test data loader...")
        return self._create_loader(
            root_path=self.path_config.testing_data_dir,
            transformer=self.train_config.basic_transformer,
            shuffle=False,
        )
