from typing import ClassVar
from dataclasses import dataclass
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
