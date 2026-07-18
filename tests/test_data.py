"""Tests for image discovery, persistent splitting, and transformations."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from applied_ai_midterm.data import (
    MANIFEST_COLUMNS,
    discover_images,
    prepare_splits,
)
from applied_ai_midterm.transforms import (
    SRGANPairTransform,
    classifier_transform,
    denormalize_classifier,
    denormalize_srgan,
)


def create_image(path: Path, color: tuple[int, int, int]) -> None:
    """Create a small RGB test image without using the real dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 40), color=color).save(path)


def build_dataset(root: Path, *, nested_train: bool = False) -> Path:
    """Build a balanced temporary binary image dataset."""
    raw_dir = root / "raw"
    class_root = raw_dir / "train" if nested_train else raw_dir
    extensions = ("jpg", "JPEG", "png", "webp")
    for class_index, class_name in enumerate(("alpha", "zeta")):
        for image_index in range(10):
            extension = extensions[image_index % len(extensions)]
            create_image(
                class_root / class_name / "nested" / f"image_{image_index}.{extension}",
                (class_index * 180, image_index * 10, 80),
            )
    (class_root / "alpha" / "ignore.txt").write_text("not an image")
    return raw_dir


@pytest.mark.parametrize("nested_train", [False, True])
def test_discover_images_supports_both_layouts(
    tmp_path: Path,
    nested_train: bool,
) -> None:
    raw_dir = build_dataset(tmp_path, nested_train=nested_train)

    manifest = discover_images(raw_dir)

    assert len(manifest) == 20
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert manifest.groupby("class_name").size().to_dict() == {
        "alpha": 10,
        "zeta": 10,
    }
    assert manifest.groupby("class_name")["label"].first().to_dict() == {
        "alpha": 0,
        "zeta": 1,
    }
    assert all(not Path(path).is_absolute() for path in manifest["filepath"])


@pytest.mark.parametrize("class_count", [1, 3])
def test_discover_images_requires_exactly_two_classes(
    tmp_path: Path,
    class_count: int,
) -> None:
    raw_dir = tmp_path / "raw"
    for index in range(class_count):
        create_image(raw_dir / f"class_{index}" / "sample.jpg", (index, 0, 0))

    with pytest.raises(ValueError, match="exactly two class directories"):
        discover_images(raw_dir)


def test_prepare_splits_is_stratified_and_persistent(tmp_path: Path) -> None:
    raw_dir = build_dataset(tmp_path)
    split_dir = tmp_path / "splits"

    train_manifest, test_manifest = prepare_splits(raw_dir, split_dir)

    assert len(train_manifest) == 14
    assert len(test_manifest) == 6
    assert train_manifest.groupby("class_name").size().to_dict() == {
        "alpha": 7,
        "zeta": 7,
    }
    assert test_manifest.groupby("class_name").size().to_dict() == {
        "alpha": 3,
        "zeta": 3,
    }
    assert list(pd.read_csv(split_dir / "train.csv").columns) == MANIFEST_COLUMNS
    assert list(pd.read_csv(split_dir / "test.csv").columns) == MANIFEST_COLUMNS
    assert not set(train_manifest["filepath"]) & set(test_manifest["filepath"])

    create_image(raw_dir / "alpha" / "new_after_split.jpg", (1, 2, 3))
    reused_train, reused_test = prepare_splits(
        raw_dir,
        split_dir,
        random_seed=999,
    )
    pd.testing.assert_frame_equal(train_manifest, reused_train)
    pd.testing.assert_frame_equal(test_manifest, reused_test)


def test_classifier_transforms_and_denormalization() -> None:
    image = Image.new("RGB", (80, 60), color=(32, 128, 224))
    evaluation_transform = classifier_transform(128, training=False)

    first = evaluation_transform(image)
    second = evaluation_transform(image)
    displayed = denormalize_classifier(first)

    assert first.shape == (3, 128, 128)
    assert torch.equal(first, second)
    assert displayed.shape == first.shape
    assert torch.all((displayed >= 0) & (displayed <= 1))


def test_srgan_pair_shapes_range_and_denormalization() -> None:
    image = Image.new("RGB", (80, 60), color=(20, 100, 200))
    pair_transform = SRGANPairTransform(32, 128, training=False)

    low_resolution, high_resolution = pair_transform(image)

    assert low_resolution.shape == (3, 32, 32)
    assert high_resolution.shape == (3, 128, 128)
    assert torch.all((low_resolution >= -1) & (low_resolution <= 1))
    assert torch.all((high_resolution >= -1) & (high_resolution <= 1))
    displayed = denormalize_srgan(high_resolution)
    assert torch.all((displayed >= 0) & (displayed <= 1))


def test_training_transforms_produce_required_shapes() -> None:
    image = Image.new("RGB", (80, 60), color=(100, 140, 180))

    classifier_image = classifier_transform(128, training=True)(image)
    low_resolution, high_resolution = SRGANPairTransform(
        32,
        128,
        training=True,
    )(image)

    assert classifier_image.shape == (3, 128, 128)
    assert low_resolution.shape == (3, 32, 32)
    assert high_resolution.shape == (3, 128, 128)
