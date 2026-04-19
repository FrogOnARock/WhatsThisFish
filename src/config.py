from dataclasses import dataclass, field
import yaml
from pathlib import Path
import logging
import os
import sys



@dataclass
class S3Config:
    base_url: str
    bucket: str
    datasets: dict[str, str]
    output_paths: dict[str, str]


@dataclass
class GCSConfig:
    bucket: str
    prefixes: dict[str, str]

@dataclass
class YoloConfig:
    data_paths: dict[str, str]

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

@dataclass
class ModelConfig:
    yolo: YoloConfig

    @classmethod
    def from_yaml(cls, path: str = "config/yolo_conifg.yaml") -> "ModelConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            yolo=YoloConfig(**raw["yolo_config"]),
        )


# Singleton — loaded once, imported everywhere
_config = None
_model_config = None

def get_config(path: str | None = None) -> AppConfig:
    global _config
    if _config is None:
        if path is None:
            path = str(Path(__file__).parent / "config" / "data_config.yaml")
        _config = AppConfig.from_yaml(path)
    return _config


def get_model_config(path: str | None = None) -> ModelConfig:
    global _model_config
    if _model_config is None:
        if path is None:
            path = str(Path(__file__).parent / "config" / "yolo_conifg.yaml")
        _model_config = ModelConfig.from_yaml(path)
    return _model_config

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

