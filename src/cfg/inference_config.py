from functools import cached_property
from typing import ClassVar
from dataclasses import dataclass, field
from src.cfg.model_config import ModelConfig
from torchvision import transforms


@dataclass(frozen=True, kw_only=True)
class InferenceConfig:
    model_name: str
    device: str
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    ALLOWED_DEVICES: ClassVar[tuple[str, str, str]] = ("cpu", "cuda", "mps")

    def __post_init__(self):
        if self.device not in InferenceConfig.ALLOWED_DEVICES:
            raise ValueError(
                f"Choose one of the allowed devices: {self.ALLOWED_DEVICES}"
            )

    @cached_property
    def basic_transforms(self) -> list:
        resizer = transforms.Resize([ModelConfig.img_size, ModelConfig.img_size])
        tensor = transforms.ToTensor()
        normalizer = transforms.Normalize(mean=self.mean, std=self.std)

        return [resizer, tensor, normalizer]

    @cached_property
    def basic_transformer(self) -> transforms.Compose:
        return transforms.Compose(self.basic_transforms)
