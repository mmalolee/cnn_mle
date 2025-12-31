from src.cfg.paths_config import PathsConfig
from src.utils import get_logger

log = get_logger("CNN-MLE")

log.info(f"Project's file is in: {PathsConfig().base_dir}")

if PathsConfig().data_dir.exists():
    log.info(f"Data folder path: {PathsConfig().data_dir}.")
    log.info(f"Raw data path: {PathsConfig().raw_data_dir}")
    log.info(f"Processed data path: {PathsConfig().processed_data_dir}")

else:
    log.error(f"Data folder has not been detected.")

log.info(f"Source code path: {PathsConfig().src_dir}")
log.info(f"Config code path: {PathsConfig().cfg_dir}")
