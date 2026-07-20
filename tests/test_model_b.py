"""Synthetic tests for generated-image classifier Model B."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from applied_ai_midterm.data import MANIFEST_COLUMNS
from applied_ai_midterm.datasets import split_training_manifest
from applied_ai_midterm.generation import (
    GENERATED_MANIFEST_COLUMNS,
    PROVENANCE_FILENAME,
)
from applied_ai_midterm.model_b import (
    create_model_b_dataloaders,
    load_model_b_manifest,
    split_model_b_manifest,
)
from applied_ai_midterm.training import fit_classifier, run_classifier_epoch


def build_model_b_data(root: Path) -> tuple[Path, Path]:
    """Create canonical source records, generated images, and provenance."""
    source_records: list[dict[str, str | int]] = []
    generated_records: list[dict[str, str | int]] = []
    generated_root = root / "generated"
    output_directory = generated_root / "model_b_train"
    for label, class_name in enumerate(("class_a", "class_b")):
        for index in range(10):
            source_filepath = f"{class_name}/source_{index}.png"
            generated_filepath = (
                Path("model_b_train") / class_name / f"generated_{index}.png"
            )
            image_path = generated_root / generated_filepath
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB", (128, 128), color=(label * 180, index * 8, 100)
            ).save(image_path)
            source_records.append(
                {
                    "filepath": source_filepath,
                    "class_name": class_name,
                    "label": label,
                }
            )
            generated_records.append(
                {
                    "filepath": generated_filepath.as_posix(),
                    "source_filepath": source_filepath,
                    "class_name": class_name,
                    "label": label,
                }
            )

    source_manifest = root / "splits" / "train.csv"
    source_manifest.parent.mkdir(parents=True)
    pd.DataFrame(source_records, columns=MANIFEST_COLUMNS).to_csv(
        source_manifest, index=False
    )
    generated_manifest = generated_root / "model_b_train.csv"
    pd.DataFrame(
        generated_records, columns=GENERATED_MANIFEST_COLUMNS
    ).to_csv(generated_manifest, index=False)
    provenance = {
        "checkpoint_identifier": "synthetic-checkpoint-sha256",
        "checkpoint_filename": "srgan_epoch_0150.pt",
        "checkpoint_epoch": 150,
        "configuration": {
            "low_resolution_size": 32,
            "high_resolution_size": 128,
            "srgan_epochs": 150,
        },
        "class_mapping": {"class_a": 0, "class_b": 1},
        "low_resolution_size": 32,
        "high_resolution_size": 128,
    }
    (output_directory / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    return generated_manifest, source_manifest


def test_model_b_manifest_loads_only_canonical_training_sources(
    tmp_path: Path,
) -> None:
    generated_manifest, source_manifest = build_model_b_data(tmp_path)

    loaded = load_model_b_manifest(generated_manifest, source_manifest)

    assert list(loaded.columns) == GENERATED_MANIFEST_COLUMNS
    assert len(loaded) == 20
    assert loaded["source_filepath"].is_unique


def test_model_b_preserves_model_a_source_partition_without_leakage(
    tmp_path: Path,
) -> None:
    generated_manifest, source_manifest = build_model_b_data(tmp_path)
    source_train, source_validation = split_training_manifest(source_manifest)

    model_b_train, model_b_validation = split_model_b_manifest(
        generated_manifest, source_manifest
    )

    assert set(model_b_train["source_filepath"]) == set(source_train["filepath"])
    assert set(model_b_validation["source_filepath"]) == set(
        source_validation["filepath"]
    )
    assert not set(model_b_train["source_filepath"]) & set(
        model_b_validation["source_filepath"]
    )


def test_model_b_rejects_nontraining_or_duplicate_source_records(
    tmp_path: Path,
) -> None:
    generated_manifest, source_manifest = build_model_b_data(tmp_path)
    records = pd.read_csv(generated_manifest)
    records.loc[0, "source_filepath"] = "reserved_test/leaked.png"
    records.to_csv(generated_manifest, index=False)

    with pytest.raises(ValueError, match="canonical training records"):
        load_model_b_manifest(generated_manifest, source_manifest)

    generated_manifest, source_manifest = build_model_b_data(tmp_path / "duplicate")
    records = pd.read_csv(generated_manifest)
    records.loc[1, "source_filepath"] = records.loc[0, "source_filepath"]
    records.to_csv(generated_manifest, index=False)
    with pytest.raises(ValueError, match="duplicate source_filepath"):
        load_model_b_manifest(generated_manifest, source_manifest)


def test_model_b_loader_and_one_synthetic_training_step(tmp_path: Path) -> None:
    generated_manifest, source_manifest = build_model_b_data(tmp_path)
    loaders = create_model_b_dataloaders(
        generated_manifest,
        source_manifest,
        batch_size=4,
        num_workers=0,
    )
    model = nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(2, 1),
    )
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    metrics = run_classifier_epoch(
        model,
        loaders.train,
        nn.BCEWithLogitsLoss(),
        torch.device("cpu"),
        optimizer=optimizer,
    )

    assert loaders.train_size == 16
    assert loaders.validation_size == 4
    assert loaders.generator_provenance["checkpoint_identifier"] == (
        "synthetic-checkpoint-sha256"
    )
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "loss",
    }
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_model_b_checkpoints_are_separate_and_record_generator(
    tmp_path: Path,
) -> None:
    images = torch.tensor([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = nn.Linear(2, 1)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    configuration = {
        "model": "classifier_b",
        "generator_checkpoint_identifier": "synthetic-checkpoint-sha256",
    }

    fit_classifier(
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
        configuration=configuration,
        checkpoint_prefix="classifier_b",
    )

    latest = tmp_path / "classifier_b_latest.pt"
    assert latest.is_file()
    assert (tmp_path / "classifier_b_best.pt").is_file()
    assert not (tmp_path / "classifier_a_latest.pt").exists()
    checkpoint = torch.load(latest, weights_only=False)
    assert checkpoint["configuration"] == configuration
    resume_model = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="data provenance"):
        fit_classifier(
            resume_model,
            loader,
            loader,
            AdamW(resume_model.parameters(), lr=1e-3),
            nn.BCEWithLogitsLoss(),
            epochs=2,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path,
            class_mapping={"class_a": 0, "class_b": 1},
            random_seed=42,
            configuration={
                **configuration,
                "generator_checkpoint_identifier": "different-generator",
            },
            resume_from=latest,
            checkpoint_prefix="classifier_b",
        )
    with pytest.raises(ValueError, match="does not belong"):
        fit_classifier(
            nn.Linear(2, 1),
            loader,
            loader,
            AdamW(nn.Linear(2, 1).parameters(), lr=1e-3),
            nn.BCEWithLogitsLoss(),
            epochs=2,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path,
            class_mapping={"class_a": 0, "class_b": 1},
            random_seed=42,
            configuration=configuration,
            resume_from=latest,
            checkpoint_prefix="classifier_a",
        )
