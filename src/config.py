from pathlib import Path
from dataclasses import dataclass


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"


@dataclass(frozen=True)
class InferenceConfig:
    """
    Class which keeps MRI prediction parameters.
    """

    model_name: str = "cnn_anti_tumor"
    image_size: tuple[int, int] = (250, 250)
    ALLOWED_DEVICES: tuple[str, str, str] = ("cpu", "cuda", "mps")
    device: str = "cpu"
    batch_size: int = 32

    def __post_init__(self):
        if self.device not in self.ALLOWED_DEVICES:
            raise ValueError(
                f"Choose one of the allowed devices: {self.ALLOWED_DEVICES}"
            )

    @property
    def logs_path(self) -> Path:
        return LOGS_DIR

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / self.model_name

    @property
    def raw_data_path(self) -> Path:
        return RAW_DATA_DIR

    @property
    def processed_data_path(self) -> Path:
        return PROCESSED_DATA_DIR


if __name__ == "__main__":
    from src.utils import get_logger

    log = get_logger("CONFIG")
    log.info(f"Project's file is in: {BASE_DIR}")
    log.info(f"Raw data is in: {RAW_DATA_DIR}")
    log.info(f"Processed data is in: {PROCESSED_DATA_DIR}")

    config = InferenceConfig()

    if DATA_DIR.exists():
        log.info("Data folder has been detected.")

    else:
        log.error(f"Data folder has not been detected.")
