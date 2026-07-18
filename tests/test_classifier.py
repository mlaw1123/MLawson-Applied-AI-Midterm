"""Synthetic tests for Model A data, model, training, and checkpoints."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from applied_ai_midterm.classifier import create_binary_mobilenet
from applied_ai_midterm.data import MANIFEST_COLUMNS
from applied_ai_midterm.datasets import (
    ManifestImageDataset,
    create_classifier_dataloaders,
    split_training_manifest,
)
from applied_ai_midterm.training import (
    binary_classification_metrics,
    fit_classifier,
    load_classifier_checkpoint,
    save_classifier_checkpoint,
    seed_everything,
    select_device,
)
from applied_ai_midterm.transforms import classifier_transform


def build_training_manifest(
    root: Path,
    images_per_class: int = 10,
) -> tuple[Path, Path]:
    """Create a temporary balanced train.csv and its RGB images."""
    raw_dir = root / "raw"
    records = []
    for label, class_name in enumerate(("class_a", "class_b")):
        for index in range(images_per_class):
            relative_path = Path(class_name) / f"image_{index}.png"
            image_path = raw_dir / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (48, 48),
                color=(label * 180, index * 8, 100),
            ).save(image_path)
            records.append(
                {
                    "filepath": relative_path.as_posix(),
                    "class_name": class_name,
                    "label": label,
                }
            )
    manifest_path = root / "train.csv"
    pd.DataFrame(records, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    return manifest_path, raw_dir


def test_validation_split_is_stratified_reproducible_and_disjoint(
    tmp_path: Path,
) -> None:
    manifest_path, _ = build_training_manifest(tmp_path)

    first_train, first_validation = split_training_manifest(manifest_path)
    second_train, second_validation = split_training_manifest(manifest_path)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_validation, second_validation)
    assert first_train.groupby("label").size().to_dict() == {0: 8, 1: 8}
    assert first_validation.groupby("label").size().to_dict() == {0: 2, 1: 2}
    assert not set(first_train["filepath"]) & set(first_validation["filepath"])


def test_validation_split_rejects_test_manifest(tmp_path: Path) -> None:
    manifest_path, _ = build_training_manifest(tmp_path)
    test_path = manifest_path.with_name("test.csv")
    manifest_path.rename(test_path)

    with pytest.raises(ValueError, match="train.csv manifest only"):
        split_training_manifest(test_path)


def test_dataloaders_use_augmented_train_and_deterministic_validation(
    tmp_path: Path,
) -> None:
    manifest_path, raw_dir = build_training_manifest(tmp_path)

    loaders = create_classifier_dataloaders(
        manifest_path,
        raw_dir,
        batch_size=4,
        validation_ratio=0.20,
        random_seed=42,
        num_workers=0,
    )

    train_images, train_labels = next(iter(loaders.train))
    validation_images, validation_labels = next(iter(loaders.validation))
    assert train_images.shape == (4, 3, 128, 128)
    assert validation_images.shape == (4, 3, 128, 128)
    assert train_labels.shape == validation_labels.shape == (4,)
    assert loaders.class_mapping == {"class_a": 0, "class_b": 1}
    assert loaders.train_size == 16
    assert loaders.validation_size == 4


def test_manifest_dataset_reports_missing_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest = pd.DataFrame(
        [["class_a/missing.png", "class_a", 0]],
        columns=MANIFEST_COLUMNS,
    )

    with pytest.raises(FileNotFoundError, match="missing image"):
        ManifestImageDataset(
            manifest,
            raw_dir,
            classifier_transform(128, training=False),
        )


def test_mobilenet_has_one_logit_and_smoke_backward() -> None:
    model = create_binary_mobilenet(pretrained=False)
    assert isinstance(model.classifier[-1], nn.Linear)
    assert model.classifier[-1].out_features == 1

    images = torch.randn(2, 3, 128, 128)
    labels = torch.tensor([0.0, 1.0])
    logits = model(images).flatten()
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    assert logits.shape == (2,)
    assert torch.isfinite(loss)
    assert model.classifier[-1].weight.grad is not None


def test_metrics_apply_sigmoid_to_logits() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.tensor([-4.0, -2.0, 2.0, 4.0])

    metrics = binary_classification_metrics(labels, logits)

    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
    }


def test_checkpoint_round_trip_restores_all_required_state(tmp_path: Path) -> None:
    seed_everything(42)
    model = nn.Linear(2, 1)
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1)
    history = {
        "train": [{"epoch": 1.0, "loss": 0.5}],
        "validation": [{"epoch": 1.0, "loss": 0.4, "f1": 0.8}],
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    original_weight = model.weight.detach().clone()

    save_classifier_checkpoint(
        checkpoint_path,
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        history=history,
        class_mapping={"class_a": 0, "class_b": 1},
        random_seed=42,
        configuration={"epochs": 20},
        best_validation_f1=0.8,
    )
    with torch.no_grad():
        model.weight.zero_()
    checkpoint = load_classifier_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert torch.equal(model.weight, original_weight)
    assert checkpoint["epoch"] == 1
    assert checkpoint["history"] == history
    assert checkpoint["class_mapping"] == {"class_a": 0, "class_b": 1}
    assert checkpoint["random_seed"] == 42
    assert checkpoint["configuration"] == {"epochs": 20}
    assert checkpoint["scheduler_state"] is not None
    assert "data_loader_state" in checkpoint
    assert checkpoint["rng_state"] is not None


def test_fit_classifier_runs_one_synthetic_epoch_and_saves_checkpoints(
    tmp_path: Path,
) -> None:
    images = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = nn.Linear(2, 1)
    optimizer = Adam(model.parameters(), lr=0.01)

    history = fit_classifier(
        model,
        loader,
        loader,
        optimizer,
        nn.BCEWithLogitsLoss(),
        epochs=1,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
        class_mapping={"class_a": 0, "class_b": 1},
        random_seed=42,
        configuration={"epochs": 1},
    )

    assert len(history["train"]) == len(history["validation"]) == 1
    assert (tmp_path / "classifier_a_latest.pt").is_file()
    assert (tmp_path / "classifier_a_best.pt").is_file()
    saved = torch.load(
        tmp_path / "classifier_a_latest.pt",
        weights_only=False,
    )
    assert saved["data_loader_state"] is None
    assert set(history["train"][0]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "loss",
        "epoch",
    }


def test_device_selection_returns_supported_device() -> None:
    assert select_device().type in {"cuda", "mps", "cpu"}
