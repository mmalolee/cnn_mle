from pathlib import Path
from dataclasses import dataclass
from torchvision import transforms
from functools import cached_property
from src.paths_config import PathsConfig


@dataclass(frozen=True)
class ModelConfig:
    img_size: int = 250
    num_classes: int = 4
    input_channels: int = 3


@dataclass(frozen=True)
class InferenceConfig:
    """
    Class which keeps MRI prediction parameters.
    """

    model_name: str = "cnn_anti_tumor"
    ALLOWED_DEVICES: tuple[str, str, str] = ("cpu", "cuda", "mps")
    device: str = "cpu"
    batch_size: int = 1

    def __post_init__(self):
        if self.device not in self.ALLOWED_DEVICES:
            raise ValueError(
                f"Choose one of the allowed devices: {self.ALLOWED_DEVICES}"
            )

    @cached_property
    def transforms(self) -> transforms.Compose:
        resizer = transforms.Resize([ModelConfig.img_size, ModelConfig.img_size])
        tensor = transforms.ToTensor()
        transformer = transforms.Compose([resizer, tensor])

        return transformer


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    checkopoint_dir: Path = PathsConfig().models_dir


if __name__ == "__main__":
    from src.utils import get_logger

    log = get_logger("CNN-MLE")
    log.info(f"Project's file is in: {PathsConfig().base_dir}")
    log.info(f"Raw data is in: {PathsConfig().raw_data_dir}")
    log.info(f"Processed data is in: {PathsConfig().processed_data_dir}")

    if PathsConfig().data_dir.exists():
        log.info("Data folder has been detected.")

    else:
        log.error(f"Data folder has not been detected.")
