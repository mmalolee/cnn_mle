from functools import cached_property
from dataclasses import dataclass
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from src.cfg.inference_config import InferenceConfig
from src.cfg.paths_config import PathsConfig
from src.cfg.training_config import TrainingConfig


@dataclass(frozen=True)
class DataLoad:
    path_config: PathsConfig
    train_config: TrainingConfig
    inference_config: InferenceConfig

    def _create_loader(self, root_path, shuffle=False) -> DataLoader:
        dataset = datasets.ImageFolder(
            root=root_path, transform=self.inference_config.transformer
        )

        cpus = os.cpu_count() or 1
        workers = cpus // 2

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
        )

    def get_train_loader(self) -> DataLoader:
        return self._create_loader(self.path_config.training_data_dir, shuffle=True)

    def get_test_loader(self) -> DataLoader:
        return self._create_loader(self.path_config.testing_data_dir, shuffle=False)

    @cached_property
    def classes(self):
        return sorted(
            [f.name for f in self.path_config.testing_data_dir.iterdir() if f.is_dir()]
        )
