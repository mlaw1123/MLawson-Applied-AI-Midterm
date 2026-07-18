"""Load and validate project configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isclose
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Validated settings shared by project scripts and notebooks."""

    random_seed: int
    train_ratio: float
    test_ratio: float
    low_resolution_size: int
    high_resolution_size: int
    classifier_batch_size: int
    srgan_batch_size: int
    classifier_epochs: int
    srgan_epochs: int
    checkpoint_interval: int
    num_workers: int

    def __post_init__(self) -> None:
        """Reject configuration values that violate the experiment contract."""
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if not 0 < self.train_ratio < 1 or not 0 < self.test_ratio < 1:
            raise ValueError("train_ratio and test_ratio must each be between 0 and 1")
        if not isclose(self.train_ratio + self.test_ratio, 1.0):
            raise ValueError("train_ratio and test_ratio must sum to 1.0")
        if self.high_resolution_size != self.low_resolution_size * 4:
            raise ValueError("high_resolution_size must be 4x low_resolution_size")

        positive_fields = (
            "low_resolution_size",
            "high_resolution_size",
            "classifier_batch_size",
            "srgan_batch_size",
            "classifier_epochs",
            "srgan_epochs",
            "checkpoint_interval",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")
        if self.srgan_epochs < 150:
            raise ValueError("srgan_epochs must be at least 150")
        if self.checkpoint_interval != 5:
            raise ValueError("checkpoint_interval must be 5 epochs")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Read a YAML configuration file and return validated project settings."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    expected_keys = {field.name for field in fields(ProjectConfig)}
    missing_keys = expected_keys - raw_config.keys()
    unexpected_keys = raw_config.keys() - expected_keys
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Configuration is missing required keys: {missing}")
    if unexpected_keys:
        unexpected = ", ".join(sorted(unexpected_keys))
        raise ValueError(f"Configuration contains unknown keys: {unexpected}")

    try:
        return ProjectConfig(**raw_config)
    except TypeError as error:
        raise ValueError(f"Configuration values have invalid types: {error}") from error

