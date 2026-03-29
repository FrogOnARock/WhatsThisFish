from dataclasses import dataclass, field
import yaml
from pathlib import Path
import logging
import os
import sys



@dataclass
class S3Config:
    bucket: str
    datasets: dict[str, str]
    output_paths: dict[str, str]


@dataclass
class GCSConfig:
    bucket: str
    prefixes: dict[str, str]
    confidence_threshold: float = 0.85


@dataclass
class AppConfig:
    s3: S3Config
    gcs: GCSConfig

    @classmethod
    def from_yaml(cls, path: str = "config/data_config.yaml") -> "AppConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            s3=S3Config(**raw["s3"]),
            gcs=GCSConfig(**raw["gcs"]),
        )


# Singleton — loaded once, imported everywhere
_config = None

def get_config(path: str = "src/config/data_config.yaml") -> AppConfig:
    global _config
    if _config is None:
        path = os.getcwd() + "/" + path
        _config = AppConfig.from_yaml(path)
    return _config

def _get_logger(name: str):
    """

    Probably should just move this up
    :return:
    """
    logging_path = Path(__file__).parents[1] / "logs" / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatting = '%(asctime)s - %(levelname)s - %(message)s'

    if logger.handlers:
        return logger

    if os.path.exists(logging_path):
        pass
    else:
        logging_path.parent.mkdir(parents=True, exist_ok=True)

    # Set file handler for output of logging
    file_handler = logging.FileHandler(logging_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(formatting))

    # Set console handler for output of logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(formatting))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

