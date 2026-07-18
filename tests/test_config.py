"""Tests for project configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from applied_ai_midterm.config import ProjectConfig, load_config

VALID_CONFIG = {
    "random_seed": 42,
    "train_ratio": 0.70,
    "test_ratio": 0.30,
    "low_resolution_size": 32,
    "high_resolution_size": 128,
    "classifier_batch_size": 32,
    "srgan_batch_size": 16,
    "classifier_epochs": 20,
    "srgan_epochs": 150,
    "checkpoint_interval": 5,
    "num_workers": 2,
}


def write_config(path: Path, values: dict[str, int | float]) -> None:
    """Write test configuration values to a temporary YAML file."""
    path.write_text(yaml.safe_dump(values), encoding="utf-8")


def test_load_config_returns_validated_dataclass(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    write_config(config_path, VALID_CONFIG)

    config = load_config(config_path)

    assert isinstance(config, ProjectConfig)
    assert config.random_seed == 42
    assert config.train_ratio == pytest.approx(0.70)
    assert config.high_resolution_size == 128
    assert config.srgan_epochs == 150


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(missing_path)


def test_load_config_reports_missing_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    incomplete = VALID_CONFIG.copy()
    incomplete.pop("random_seed")
    write_config(config_path, incomplete)

    with pytest.raises(ValueError, match="missing required keys: random_seed"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"train_ratio": 0.8}, "must sum to 1.0"),
        ({"high_resolution_size": 64}, "must be 4x"),
        ({"srgan_epochs": 149}, "must be at least 150"),
        ({"checkpoint_interval": 10}, "must be 5 epochs"),
        ({"num_workers": -1}, "must be non-negative"),
    ],
)
def test_load_config_rejects_invalid_values(
    tmp_path: Path,
    changes: dict[str, int | float],
    message: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    invalid = VALID_CONFIG | changes
    write_config(config_path, invalid)

    with pytest.raises(ValueError, match=message):
        load_config(config_path)

