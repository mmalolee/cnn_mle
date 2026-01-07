from pathlib import Path
from functools import cached_property
from dataclasses import dataclass
from src.cfg.paths_config import PathsConfig
from src.cfg.inference_config import InferenceConfig
from torchvision import transforms


@dataclass(frozen=True, kw_only=True)
class TrainingConfig(InferenceConfig):
    epochs: int
    learning_rate: float
    batch_size: int
    checkpoint_dir: Path = PathsConfig().models_dir

    def __post_init__(self):
        super().__post_init__()

        if self.epochs < 1:
            raise ValueError("You need at least 1 epoch to train.")

        if not (0 < self.learning_rate < 1):
            raise ValueError(
                f"Learning rate out of range: {self.learning_rate}. Number between 0 and 1 required"
            )

        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")

    @cached_property
    def training_transforms(self) -> list:
        basics = self.basic_transforms

        return [basics[0], transforms.RandomRotation(10, fill=0), *basics[1:]]

    @cached_property
    def training_transformer(self) -> transforms.Compose:
        return transforms.Compose(self.training_transforms)
