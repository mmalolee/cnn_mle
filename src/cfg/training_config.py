from pathlib import Path
from dataclasses import dataclass
from src.cfg.paths_config import PathsConfig
from src.cfg.inference_config import InferenceConfig
from torchvision import transforms


@dataclass(frozen=True, kw_only=True)
class TrainingConfig(InferenceConfig):
    epochs: int
    learning_rate: float
    batch_size: int
    rotation: int = 10
    checkpoint_dir: Path = PathsConfig().checkpoints_dir

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

    @property
    def training_transforms(self) -> list:
        t_transforms = transforms.RandomRotation(self.rotation)
        return [t_transforms]

    @property
    def training_transformer(self) -> transforms.Compose:
        return transforms.Compose(
            self.pil_transforms + self.training_transforms + self.tensor_transforms
        )
