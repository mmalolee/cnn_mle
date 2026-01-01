from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    img_size: int
    num_classes: int
    input_channels: int
