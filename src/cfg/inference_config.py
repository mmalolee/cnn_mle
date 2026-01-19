from typing import ClassVar
from dataclasses import dataclass, field
from torchvision import transforms
from src.cfg.model_config import ModelConfig


@dataclass(frozen=True, kw_only=True)
class InferenceConfig:
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    device: str
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    ALLOWED_DEVICES: ClassVar[tuple[str, str, str]] = ("cpu", "cuda", "mps")

    def __post_init__(self):
        if self.device not in InferenceConfig.ALLOWED_DEVICES:
            raise ValueError(
                f"Choose one of the allowed devices: {self.ALLOWED_DEVICES}"
            )

        if len(self.mean) != 3:
            raise ValueError(
                f"Mean must have exactly 3 values for RGB but {len(self.mean)} were given."
            )

        if len(self.std) != 3:
            raise ValueError(
                f"Std must have exactly 3 values for RGB but {len(self.std)} were given."
            )

        if any(s <= 0 for s in self.std):
            raise ValueError("Std must be greater than 0 to avoid division by zero.")

    @property
    def img_size(self) -> int:
        return 250

    @property
    def pil_transforms(self) -> list:
        return [transforms.Resize([self.img_size, self.img_size])]

    @property
    def tensor_transforms(self) -> list:
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

    @property
    def basic_transformer(self) -> transforms.Compose:
        return transforms.Compose(self.pil_transforms + self.tensor_transforms)
