"""Discover binary image data and persist one reproducible train/test split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

VALID_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MANIFEST_COLUMNS = ["filepath", "class_name", "label"]


def find_class_directories(raw_dir: str | Path) -> tuple[Path, Path]:
    """Return the two class directories from either supported raw-data layout."""
    root = Path(raw_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {root}")

    class_root = root / "train" if (root / "train").is_dir() else root
    class_dirs = sorted(
        (
            path
            for path in class_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.name.casefold(),
    )
    if len(class_dirs) != 2:
        names = ", ".join(path.name for path in class_dirs) or "none"
        raise ValueError(
            "Binary classification requires exactly two class directories under "
            f"{class_root}; found {len(class_dirs)} ({names})"
        )
    return class_dirs[0], class_dirs[1]


def discover_images(raw_dir: str | Path) -> pd.DataFrame:
    """Recursively discover supported images and assign stable binary labels."""
    root = Path(raw_dir).expanduser()
    class_dirs = find_class_directories(root)
    records: list[dict[str, str | int]] = []

    for label, class_dir in enumerate(class_dirs):
        images = sorted(
            (
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.casefold() in VALID_IMAGE_EXTENSIONS
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        if not images:
            raise ValueError(
                f"Class directory contains no supported images: {class_dir}"
            )
        records.extend(
            {
                "filepath": path.relative_to(root).as_posix(),
                "class_name": class_dir.name,
                "label": label,
            }
            for path in images
        )

    return pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)


def load_split_manifest(path: str | Path) -> pd.DataFrame:
    """Load a split manifest and validate its schema and binary label mapping."""
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            f"Split manifest must contain columns {MANIFEST_COLUMNS}: {manifest_path}"
        )
    if manifest.empty:
        raise ValueError(f"Split manifest is empty: {manifest_path}")
    if manifest["filepath"].duplicated().any():
        raise ValueError(
            f"Split manifest contains duplicate filepaths: {manifest_path}"
        )
    if set(manifest["label"].unique()) - {0, 1}:
        raise ValueError(f"Split manifest labels must be 0 or 1: {manifest_path}")

    mappings = manifest[["class_name", "label"]].drop_duplicates()
    if len(mappings) != 2 or mappings["class_name"].nunique() != 2:
        raise ValueError(
            f"Split manifest must describe exactly two class mappings: {manifest_path}"
        )
    return manifest


def prepare_splits(
    raw_dir: str | Path,
    split_dir: str | Path,
    *,
    train_ratio: float = 0.70,
    random_seed: int = 42,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create or reuse the single persisted stratified dataset split.

    Filepaths are stored relative to ``raw_dir`` so manifests remain portable
    between local development and Google Colab. Existing paired manifests are
    reused by default, preventing later experiments from silently reshuffling
    the dataset.
    """
    destination = Path(split_dir).expanduser()
    train_path = destination / "train.csv"
    test_path = destination / "test.csv"

    if train_path.exists() != test_path.exists() and not force:
        raise FileNotFoundError(
            "Only one split manifest exists. Restore both train.csv and test.csv, "
            "or rerun with force=True to deliberately replace the split."
        )
    if train_path.is_file() and test_path.is_file() and not force:
        train_manifest = load_split_manifest(train_path)
        test_manifest = load_split_manifest(test_path)
        overlap = set(train_manifest["filepath"]) & set(test_manifest["filepath"])
        if overlap:
            raise ValueError("Train and test manifests contain overlapping filepaths")
        return train_manifest, test_manifest

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    all_images = discover_images(raw_dir)
    try:
        train_manifest, test_manifest = train_test_split(
            all_images,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=all_images["label"],
        )
    except ValueError as error:
        raise ValueError(
            "Unable to create a stratified split; each class needs enough images"
        ) from error

    train_manifest = train_manifest.sort_values("filepath").reset_index(drop=True)
    test_manifest = test_manifest.sort_values("filepath").reset_index(drop=True)
    destination.mkdir(parents=True, exist_ok=True)
    train_manifest.to_csv(train_path, index=False)
    test_manifest.to_csv(test_path, index=False)
    return train_manifest, test_manifest
