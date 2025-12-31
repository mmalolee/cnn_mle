from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    img_size: int = 250
    num_classes: int = 4
    input_channels: int = 3
