"""Leakage-safe generated-image data loading for classifier Model B."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from applied_ai_midterm.data import MANIFEST_COLUMNS, load_split_manifest
from applied_ai_midterm.datasets import ManifestImageDataset, split_training_manifest
from applied_ai_midterm.generation import (
    GENERATED_MANIFEST_COLUMNS,
    PROVENANCE_FILENAME,
)
from applied_ai_midterm.transforms import classifier_transform


@dataclass(frozen=True, slots=True)
class ModelBDataLoaders:
    """Model B loaders plus the generator provenance used to create inputs."""

    train: DataLoader[tuple[Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor]]
    class_mapping: dict[str, int]
    train_size: int
    validation_size: int
    generator_provenance: dict[str, Any]


def load_model_b_manifest(
    generated_manifest_path: str | Path,
    source_train_manifest_path: str | Path,
) -> pd.DataFrame:
    """Validate generated records against the canonical original train split."""
    generated_path = Path(generated_manifest_path).expanduser()
    if generated_path.name != "model_b_train.csv":
        raise ValueError(
            "Model B must consume data/generated/model_b_train.csv only"
        )
    if not generated_path.is_file():
        raise FileNotFoundError(
            f"Generated Model B manifest not found: {generated_path}"
        )
    source_path = Path(source_train_manifest_path).expanduser()
    if source_path.name != "train.csv":
        raise ValueError("Model B source validation must use train.csv only")

    generated = pd.read_csv(generated_path)
    if list(generated.columns) != GENERATED_MANIFEST_COLUMNS:
        raise ValueError(
            "Generated manifest must contain columns "
            f"{GENERATED_MANIFEST_COLUMNS}: {generated_path}"
        )
    if generated.empty:
        raise ValueError(f"Generated manifest is empty: {generated_path}")
    for column in ("filepath", "source_filepath"):
        if generated[column].duplicated().any():
            raise ValueError(f"Generated manifest contains duplicate {column} values")
        for value in generated[column]:
            _validate_relative_path(str(value), column)
    if set(generated["label"].unique()) != {0, 1}:
        raise ValueError("Generated manifest labels must contain binary values 0 and 1")

    source = load_split_manifest(source_path)
    if set(generated["source_filepath"]) != set(source["filepath"]):
        unexpected = set(generated["source_filepath"]) - set(source["filepath"])
        missing = set(source["filepath"]) - set(generated["source_filepath"])
        raise ValueError(
            "Generated manifest must represent exactly the canonical training "
            f"records; unexpected={len(unexpected)}, missing={len(missing)}"
        )
    source_labels = source.set_index("filepath")[["class_name", "label"]]
    for row in generated.itertuples(index=False):
        expected = source_labels.loc[row.source_filepath]
        if str(row.class_name) != str(expected["class_name"]) or int(
            row.label
        ) != int(expected["label"]):
            raise ValueError(
                "Generated class or label does not match its source record: "
                f"{row.source_filepath}"
            )
    return generated


def split_model_b_manifest(
    generated_manifest_path: str | Path,
    source_train_manifest_path: str | Path,
    *,
    validation_ratio: float = 0.20,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Model A's source split to generated records without source leakage."""
    generated = load_model_b_manifest(
        generated_manifest_path, source_train_manifest_path
    )
    source_training, source_validation = split_training_manifest(
        source_train_manifest_path,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )
    training_sources = set(source_training["filepath"])
    validation_sources = set(source_validation["filepath"])
    if training_sources & validation_sources:
        raise RuntimeError("Source train and validation partitions overlap")

    training = generated[
        generated["source_filepath"].isin(training_sources)
    ].copy()
    validation = generated[
        generated["source_filepath"].isin(validation_sources)
    ].copy()
    if set(training["source_filepath"]) & set(validation["source_filepath"]):
        raise RuntimeError("Generated train and validation source records overlap")
    if len(training) + len(validation) != len(generated):
        raise ValueError("Some generated records were not assigned by the source split")
    return (
        training.sort_values("source_filepath").reset_index(drop=True),
        validation.sort_values("source_filepath").reset_index(drop=True),
    )


def load_generator_provenance(
    generated_manifest_path: str | Path,
    manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Load the immutable generator identity recorded during image generation."""
    manifest_path = Path(generated_manifest_path).expanduser()
    records = manifest if manifest is not None else pd.read_csv(manifest_path)
    roots = {Path(str(filepath)).parts[0] for filepath in records["filepath"]}
    if len(roots) != 1:
        raise ValueError("Generated image paths must share one output directory")
    metadata_path = manifest_path.parent / roots.pop() / PROVENANCE_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Generator provenance not found: {metadata_path}")
    try:
        provenance = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid generator provenance: {metadata_path}") from error
    required = {
        "checkpoint_identifier",
        "checkpoint_filename",
        "checkpoint_epoch",
        "configuration",
        "class_mapping",
        "low_resolution_size",
        "high_resolution_size",
    }
    missing = required - provenance.keys()
    if missing:
        raise ValueError(f"Generator provenance is missing keys: {sorted(missing)}")
    if not isinstance(provenance["checkpoint_identifier"], str) or not provenance[
        "checkpoint_identifier"
    ]:
        raise ValueError("Generator checkpoint identifier must be a non-empty string")
    if not isinstance(provenance["configuration"], dict) or not isinstance(
        provenance["class_mapping"], dict
    ):
        raise ValueError("Generator configuration and class mapping must be objects")
    return dict(provenance)


def create_model_b_dataloaders(
    generated_manifest_path: str | Path,
    source_train_manifest_path: str | Path,
    *,
    image_size: int = 128,
    batch_size: int = 32,
    validation_ratio: float = 0.20,
    random_seed: int = 42,
    num_workers: int = 2,
) -> ModelBDataLoaders:
    """Build Model B loaders using the exact source-level Model A partition."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    generated_path = Path(generated_manifest_path).expanduser()
    training, validation = split_model_b_manifest(
        generated_path,
        source_train_manifest_path,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )
    all_records = pd.concat([training, validation], ignore_index=True)
    provenance = load_generator_provenance(generated_path, all_records)
    mapping_rows = all_records[["class_name", "label"]].drop_duplicates()
    class_mapping = {
        str(row.class_name): int(row.label)
        for row in mapping_rows.itertuples(index=False)
    }
    if class_mapping != {
        str(name): int(label)
        for name, label in provenance["class_mapping"].items()
    }:
        raise ValueError("Generator provenance class mapping does not match manifest")

    image_root = generated_path.parent
    training_dataset = ManifestImageDataset(
        training[MANIFEST_COLUMNS],
        image_root,
        classifier_transform(image_size, training=True),
    )
    validation_dataset = ManifestImageDataset(
        validation[MANIFEST_COLUMNS],
        image_root,
        classifier_transform(image_size, training=False),
    )
    generator = torch.Generator().manual_seed(random_seed)
    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=generator,
        **common_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_options,
    )
    return ModelBDataLoaders(
        train=train_loader,
        validation=validation_loader,
        class_mapping=class_mapping,
        train_size=len(training_dataset),
        validation_size=len(validation_dataset),
        generator_provenance=provenance,
    )


def _validate_relative_path(value: str, column: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{column} must contain safe relative paths: {value}")


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
