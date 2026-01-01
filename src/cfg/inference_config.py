from functools import cached_property
from dataclasses import dataclass
from src.cfg.model_config import ModelConfig
from torchvision import transforms


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    device: str
    ALLOWED_DEVICES: tuple[str, str, str] = ("cpu", "cuda", "mps")

    def __post_init__(self):
        if self.device not in self.ALLOWED_DEVICES:
            raise ValueError(
                f"Choose one of the allowed devices: {self.ALLOWED_DEVICES}"
            )

    @cached_property
    def transformer(self) -> transforms.Compose:
        resizer = transforms.Resize([ModelConfig.img_size, ModelConfig.img_size])
        tensor = transforms.ToTensor()
        transformer = transforms.Compose([resizer, tensor])

        return transformer
