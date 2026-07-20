"""Synthetic tests for aligned final Model A versus Model B evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as torch_functional

from applied_ai_midterm.data import MANIFEST_COLUMNS
from applied_ai_midterm.evaluation import (
    calculate_classification_evaluation,
    evaluate_model_comparison,
)


class TinyEvaluationGenerator(nn.Module):
    """Parameter-bearing 4x generator for no-download evaluation tests."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: Tensor) -> Tensor:
        enlarged = torch_functional.interpolate(
            inputs, scale_factor=4, mode="nearest"
        )
        return (enlarged * self.scale).clamp(-1.0, 1.0)


class TinyEvaluationClassifier(nn.Module):
    """Classify dark and bright synthetic images by normalized mean."""

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.mean(dim=(1, 2, 3), keepdim=True).flatten(1)


def generator_factory(configuration: dict[str, Any]) -> nn.Module:
    assert configuration["high_resolution_size"] == 128
    return TinyEvaluationGenerator()


def classifier_factory() -> nn.Module:
    return TinyEvaluationClassifier()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_evaluation_inputs(root: Path) -> dict[str, Path]:
    """Create test records and compatible synthetic best checkpoints."""
    raw_directory = root / "raw"
    records: list[dict[str, str | int]] = []
    for label, class_name in enumerate(("dark", "bright")):
        for index in range(2):
            relative = Path(class_name) / f"image_{index}.png"
            image_path = raw_directory / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            value = 0 if label == 0 else 255
            Image.new("RGB", (45, 50), color=(value, value, value)).save(image_path)
            records.append(
                {
                    "filepath": relative.as_posix(),
                    "class_name": class_name,
                    "label": label,
                }
            )
    test_manifest = root / "splits" / "test.csv"
    test_manifest.parent.mkdir(parents=True)
    pd.DataFrame(records, columns=MANIFEST_COLUMNS).to_csv(test_manifest, index=False)

    generator_checkpoint = root / "srgan_epoch_0150.pt"
    torch.save(
        {
            "epoch": 150,
            "generator_state": TinyEvaluationGenerator().state_dict(),
            "configuration": {
                "low_resolution_size": 32,
                "high_resolution_size": 128,
                "srgan_epochs": 150,
                "residual_blocks": 1,
            },
            "class_mapping": {"dark": 0, "bright": 1},
        },
        generator_checkpoint,
    )
    generator_identifier = sha256(generator_checkpoint)

    classifier_a = root / "classifier_a_best.pt"
    classifier_b = root / "classifier_b_best.pt"
    common = {
        "epoch": 20,
        "model_state": TinyEvaluationClassifier().state_dict(),
        "class_mapping": {"dark": 0, "bright": 1},
        "best_validation_f1": 1.0,
    }
    torch.save({**common, "configuration": {"model": "classifier_a"}}, classifier_a)
    torch.save(
        {
            **common,
            "configuration": {
                "model": "classifier_b",
                "generator_provenance": {
                    "checkpoint_identifier": generator_identifier
                },
            },
        },
        classifier_b,
    )
    return {
        "raw_directory": raw_directory,
        "test_manifest": test_manifest,
        "generator_checkpoint": generator_checkpoint,
        "classifier_a": classifier_a,
        "classifier_b": classifier_b,
    }


def test_classification_metrics_reports_matrix_and_roc() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.tensor([-4.0, 3.0, 2.0, 4.0])

    result = calculate_classification_evaluation(
        labels,
        logits,
        ["zero", "one", "two", "three"],
        {"negative": 0, "positive": 1},
    )

    assert result.metrics == {
        "accuracy": 0.75,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": 0.8,
        "roc_auc": 0.75,
    }
    assert result.confusion_matrix == [[1, 1], [0, 2]]
    assert set(result.classification_report) >= {
        "negative",
        "positive",
        "accuracy",
    }
    assert result.false_positive_rate[0] == 0.0
    assert result.true_positive_rate[-1] == 1.0


def test_models_use_identical_aligned_test_records_and_save_reports(
    tmp_path: Path,
) -> None:
    paths = build_evaluation_inputs(tmp_path)
    metrics_directory = tmp_path / "reports" / "metrics"
    figures_directory = tmp_path / "reports" / "figures"

    artifacts = evaluate_model_comparison(
        test_manifest_path=paths["test_manifest"],
        raw_directory=paths["raw_directory"],
        classifier_a_checkpoint=paths["classifier_a"],
        classifier_b_checkpoint=paths["classifier_b"],
        generator_checkpoint=paths["generator_checkpoint"],
        metrics_directory=metrics_directory,
        figures_directory=figures_directory,
        batch_size=2,
        num_workers=0,
        device="cpu",
        classifier_factory=classifier_factory,
        generator_factory=generator_factory,
    )

    model_a = artifacts.evaluations["model_a_original"]
    model_b = artifacts.evaluations["model_b_srgan"]
    assert model_a.filepaths == model_b.filepaths == sorted(model_a.filepaths)
    assert model_a.labels == model_b.labels
    assert len(model_a.filepaths) == 4
    assert artifacts.comparison_table["evaluation"].tolist() == [
        "model_a_original",
        "model_b_srgan",
        "model_b_original_secondary",
    ]
    assert artifacts.ssim == pytest.approx(1.0, abs=1e-5)
    assert artifacts.psnr == float("inf")
    for output_path in (
        artifacts.metrics_json,
        artifacts.comparison_csv,
        artifacts.predictions_csv,
        artifacts.confusion_figure,
        artifacts.roc_figure,
    ):
        assert output_path.is_file()
    predictions = pd.read_csv(artifacts.predictions_csv)
    assert predictions["filepath"].tolist() == model_a.filepaths


def test_model_b_generator_checkpoint_alignment_is_enforced(tmp_path: Path) -> None:
    paths = build_evaluation_inputs(tmp_path)
    checkpoint = torch.load(paths["classifier_b"], weights_only=False)
    checkpoint["configuration"]["generator_provenance"][
        "checkpoint_identifier"
    ] = "different-generator"
    torch.save(checkpoint, paths["classifier_b"])

    with pytest.raises(ValueError, match="different generator checkpoint"):
        evaluate_model_comparison(
            test_manifest_path=paths["test_manifest"],
            raw_directory=paths["raw_directory"],
            classifier_a_checkpoint=paths["classifier_a"],
            classifier_b_checkpoint=paths["classifier_b"],
            generator_checkpoint=paths["generator_checkpoint"],
            metrics_directory=tmp_path / "metrics",
            figures_directory=tmp_path / "figures",
            batch_size=2,
            num_workers=0,
            classifier_factory=classifier_factory,
            generator_factory=generator_factory,
        )
