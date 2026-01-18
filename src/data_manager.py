from dataclasses import dataclass
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig


@dataclass(frozen=True)
class LoaderData:
    path_config: PathsConfig
    train_config: TrainingConfig

    def _create_loader(
        self, root_path: Path, transformer: transforms.Compose, shuffle: bool
    ) -> DataLoader:
        if not root_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {root_path}.")

        dataset = datasets.ImageFolder(root=root_path, transform=transformer)

        cpus = os.cpu_count()
        workers = (cpus // 2) if cpus is not None else 1

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
        )

    def get_training_data_loader(self) -> DataLoader:
        return self._create_loader(
            root_path=self.path_config.training_data_dir,
            transformer=self.train_config.training_transformer,
            shuffle=True,
        )

    def get_test_data_loader(self) -> DataLoader:
        return self._create_loader(
            root_path=self.path_config.testing_data_dir,
            transformer=self.train_config.basic_transformer,
            shuffle=False,
        )
