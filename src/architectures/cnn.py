from src.cfg.model_config import ModelConfig
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg

        self.features = nn.Sequential(
            self._conv_block(model_cfg.input_channels, 16),
            self._conv_block(16, 32),
            self._conv_block(32, 64),
        )

        self.classification = nn.Sequential(
            nn.Flatten(),
            self._linear_block(64 * 31 * 31, 128),
            self._linear_block(128, model_cfg.num_classes),
        )

    def _conv_block(self, input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _linear_block(self, input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_channels, output_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classification(x)
