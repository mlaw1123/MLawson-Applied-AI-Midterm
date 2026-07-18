"""Typed datasets and reproducible loaders for classifier training."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from applied_ai_midterm.data import MANIFEST_COLUMNS, load_split_manifest
from applied_ai_midterm.transforms import classifier_transform


class ManifestImageDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load RGB images and binary labels from a validated manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        raw_dir: str | Path,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        if list(manifest.columns) != MANIFEST_COLUMNS:
            raise ValueError(f"Manifest must contain columns {MANIFEST_COLUMNS}")
        if manifest.empty:
            raise ValueError("Manifest must contain at least one image")

        self.manifest = manifest.reset_index(drop=True).copy()
        self.raw_dir = Path(raw_dir).expanduser()
        self.transform = transform
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Raw image directory not found: {self.raw_dir}")

        missing = [
            self.raw_dir / filepath
            for filepath in self.manifest["filepath"]
            if not (self.raw_dir / filepath).is_file()
        ]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(
                f"Manifest references {len(missing)} missing image(s): {preview}"
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.manifest.iloc[index]
        image_path = self.raw_dir / str(row["filepath"])
        try:
            with Image.open(image_path) as image:
                image_tensor = self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Unable to read image: {image_path}") from error
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image_tensor, label


@dataclass(frozen=True, slots=True)
class ClassifierDataLoaders:
    """Train/validation loaders and metadata derived from the training manifest."""

    train: DataLoader[tuple[Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor]]
    class_mapping: dict[str, int]
    train_size: int
    validation_size: int


def split_training_manifest(
    train_manifest_path: str | Path,
    *,
    validation_ratio: float = 0.20,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified validation subset from ``train.csv`` only."""
    manifest_path = Path(train_manifest_path).expanduser()
    if manifest_path.name != "train.csv":
        raise ValueError(
            "Classifier training must consume the persisted train.csv manifest only"
        )
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    manifest = load_split_manifest(manifest_path)
    try:
        training, validation = train_test_split(
            manifest,
            test_size=validation_ratio,
            random_state=random_seed,
            stratify=manifest["label"],
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create a stratified validation split from train.csv"
        ) from error

    training = training.sort_values("filepath").reset_index(drop=True)
    validation = validation.sort_values("filepath").reset_index(drop=True)
    return training, validation


def create_classifier_dataloaders(
    train_manifest_path: str | Path,
    raw_dir: str | Path,
    *,
    image_size: int = 128,
    batch_size: int = 32,
    validation_ratio: float = 0.20,
    random_seed: int = 42,
    num_workers: int = 2,
) -> ClassifierDataLoaders:
    """Build seeded Model A loaders without consulting the reserved test split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    training, validation = split_training_manifest(
        train_manifest_path,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )
    mapping_rows = pd.concat([training, validation])[
        ["class_name", "label"]
    ].drop_duplicates()
    class_mapping = {
        str(row.class_name): int(row.label)
        for row in mapping_rows.itertuples(index=False)
    }
    if len(class_mapping) != 2 or set(class_mapping.values()) != {0, 1}:
        raise ValueError("Training manifest must define one unique label per class")

    training_dataset = ManifestImageDataset(
        training,
        raw_dir,
        classifier_transform(image_size, training=True),
    )
    validation_dataset = ManifestImageDataset(
        validation,
        raw_dir,
        classifier_transform(image_size, training=False),
    )
    generator = torch.Generator().manual_seed(random_seed)
    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        # Recreate seeded workers at each epoch boundary. This makes the saved
        # DataLoader generator state sufficient for exact resume ordering.
        "persistent_workers": False,
    }
    training_loader = DataLoader(
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
    return ClassifierDataLoaders(
        train=training_loader,
        validation=validation_loader,
        class_mapping=class_mapping,
        train_size=len(training_dataset),
        validation_size=len(validation_dataset),
    )


def _seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
