from pathlib import Path
from dataclasses import dataclass
from src.cfg.paths_config import PathsConfig


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    checkopoint_dir: Path = PathsConfig().models_dir
