import logging
import sys
from src.config import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(
            filename=LOGS_DIR / "log.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    log = get_logger("TEŚCIWO TEGO TYPU TUTAJ")
    log.info("informacjone")
    log.warning("warningcjon")
    log.error("errocjone")
    log.critical("kritikal")
