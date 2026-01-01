from pathlib import Path
from dataclasses import dataclass
from src.cfg.paths_config import PathsConfig


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    checkopoint_dir: Path = PathsConfig().models_dir
