from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "CNN"
    num_classes: int = 4
    input_channels: int = 3
