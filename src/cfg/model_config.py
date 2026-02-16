from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 4
    input_channels: int = 3
