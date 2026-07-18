"""Paired datasets and reproducible loaders for SRGAN training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from applied_ai_midterm.data import MANIFEST_COLUMNS
from applied_ai_midterm.datasets import split_training_manifest
from applied_ai_midterm.transforms import SRGANPairTransform


class SRGANManifestDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load aligned low/high-resolution tensors from a manifest subset."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        raw_dir: str | Path,
        transform: SRGANPairTransform,
    ) -> None:
        if list(manifest.columns) != MANIFEST_COLUMNS:
            raise ValueError(f"Manifest must contain columns {MANIFEST_COLUMNS}")
        if manifest.empty:
            raise ValueError("SRGAN manifest must contain at least one image")
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
                f"SRGAN manifest references {len(missing)} missing image(s): {preview}"
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.manifest.iloc[index]
        image_path = self.raw_dir / str(row["filepath"])
        try:
            with Image.open(image_path) as image:
                return self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Unable to read SRGAN image: {image_path}") from error


@dataclass(frozen=True, slots=True)
class SRGANDataLoaders:
    """SRGAN loaders, class metadata, and fixed comparison samples."""

    train: DataLoader[tuple[Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor]]
    class_mapping: dict[str, int]
    train_size: int
    validation_size: int
    fixed_low_resolution: Tensor
    fixed_high_resolution: Tensor


def create_srgan_dataloaders(
    train_manifest_path: str | Path,
    raw_dir: str | Path,
    *,
    low_resolution_size: int = 32,
    high_resolution_size: int = 128,
    batch_size: int = 16,
    validation_ratio: float = 0.20,
    random_seed: int = 42,
    num_workers: int = 2,
    fixed_sample_count: int = 4,
) -> SRGANDataLoaders:
    """Build paired loaders from ``train.csv`` without reading reserved tests."""
    if batch_size <= 0 or fixed_sample_count <= 0:
        raise ValueError("batch_size and fixed_sample_count must be greater than zero")
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

    training_dataset = SRGANManifestDataset(
        training,
        raw_dir,
        SRGANPairTransform(
            low_resolution_size,
            high_resolution_size,
            training=True,
        ),
    )
    validation_dataset = SRGANManifestDataset(
        validation,
        raw_dir,
        SRGANPairTransform(
            low_resolution_size,
            high_resolution_size,
            training=False,
        ),
    )
    sample_count = min(fixed_sample_count, len(validation_dataset))
    fixed_pairs = [validation_dataset[index] for index in range(sample_count)]
    fixed_low_resolution = torch.stack([pair[0] for pair in fixed_pairs])
    fixed_high_resolution = torch.stack([pair[1] for pair in fixed_pairs])

    generator = torch.Generator().manual_seed(random_seed)
    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
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
    return SRGANDataLoaders(
        train=training_loader,
        validation=validation_loader,
        class_mapping=class_mapping,
        train_size=len(training_dataset),
        validation_size=len(validation_dataset),
        fixed_low_resolution=fixed_low_resolution,
        fixed_high_resolution=fixed_high_resolution,
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
