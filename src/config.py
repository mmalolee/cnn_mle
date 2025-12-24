from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "checkpoint"


@dataclass(frozen=True)
class InferneceConfigL:
    """
    Class which keeps MRI predictions parameters.
    """

    model_name: str = "cnn_anti_tumor"
    image_size: tuple = (250, 250)
    device: str = "cpu"
    batch_size: int = 32

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
    print(f"Project's file is in: {BASE_DIR}")
    print(f"Raw data is in: {RAW_DATA_DIR}")
    print(f"Processed data is in: {PROCESSED_DATA_DIR}")

    if DATA_DIR.exists():
        print("Data folder has been detected.")

    else:
        print(f"ERROR: Data folder has not been detected.")
