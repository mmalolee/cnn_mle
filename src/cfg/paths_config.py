from pathlib import Path


class PathsConfig:
    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def training_data_dir(self) -> Path:
        return self.data_dir / "Training"

    @property
    def testing_data_dir(self) -> Path:
        return self.data_dir / "Testing"

    @property
    def checkpoints_dir(self) -> Path:
        return self.base_dir / "models" / "checkpoints"

    @property
    def models_dir(self) -> Path:
        return self.checkpoints_dir.parent

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def src_dir(self) -> Path:
        return self.base_dir / "src"

    @property
    def cfg_dir(self):
        return self.src_dir / "cfg"
