from pathlib import Path
from functools import cached_property


class PathsConfig:
    @cached_property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @cached_property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @cached_property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @cached_property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @cached_property
    def models_dir(self) -> Path:
        return self.base_dir / "models" / "checkpoints"

    @cached_property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @cached_property
    def src_dir(self) -> Path:
        return self.base_dir / "src"

    @cached_property
    def cfg_dir(self):
        return self.src_dir / "cfg"
