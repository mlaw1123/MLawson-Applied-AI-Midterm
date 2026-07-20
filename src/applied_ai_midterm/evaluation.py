"""Aligned reserved-test evaluation for classifier Models A and B."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from applied_ai_midterm.classifier import create_binary_mobilenet
from applied_ai_midterm.data import load_split_manifest
from applied_ai_midterm.generation import GeneratorFactory, load_generator_checkpoint
from applied_ai_midterm.srgan_training import (
    peak_signal_to_noise_ratio,
    structural_similarity,
)
from applied_ai_midterm.training import binary_classification_metrics
from applied_ai_midterm.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SRGANPairTransform,
    classifier_transform,
)

ClassifierFactory = Callable[[], nn.Module]


@dataclass(frozen=True, slots=True)
class ClassificationEvaluation:
    """Metrics, report data, and aligned predictions for one classifier input."""

    metrics: dict[str, float]
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    false_positive_rate: list[float]
    true_positive_rate: list[float]
    thresholds: list[float]
    filepaths: list[str]
    labels: list[int]
    probabilities: list[float]
    predictions: list[int]


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    """In-memory results and paths written by the comparison evaluation."""

    comparison_table: pd.DataFrame
    evaluations: dict[str, ClassificationEvaluation]
    psnr: float
    ssim: float
    metrics_json: Path
    comparison_csv: Path
    predictions_csv: Path
    confusion_figure: Path
    roc_figure: Path


class TestRecordDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, Tensor, str]]
):
    """Return deterministic, aligned classifier and SRGAN views of test records."""

    def __init__(
        self,
        test_manifest_path: str | Path,
        raw_directory: str | Path,
    ) -> None:
        manifest_path = Path(test_manifest_path).expanduser()
        if manifest_path.name != "test.csv":
            raise ValueError("Final evaluation must consume data/splits/test.csv")
        self.manifest = load_split_manifest(manifest_path).sort_values(
            "filepath"
        ).reset_index(drop=True)
        self.raw_directory = Path(raw_directory).expanduser()
        if not self.raw_directory.is_dir():
            raise FileNotFoundError(
                f"Raw test image directory not found: {self.raw_directory}"
            )
        for filepath in self.manifest["filepath"]:
            relative = Path(str(filepath))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Test filepath must be relative: {filepath}")
            if not (self.raw_directory / relative).is_file():
                raise FileNotFoundError(
                    f"Reserved test image not found: {self.raw_directory / relative}"
                )
        self.classifier_transform = classifier_transform(128, training=False)
        self.srgan_transform = SRGANPairTransform(32, 128, training=False)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
        row = self.manifest.iloc[index]
        filepath = str(row["filepath"])
        image_path = self.raw_directory / filepath
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                classifier_input = self.classifier_transform(image)
                low_resolution, high_resolution = self.srgan_transform(image)
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(
                f"Unable to read reserved test image: {image_path}"
            ) from error
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return (
            classifier_input,
            low_resolution,
            high_resolution,
            label,
            filepath,
        )


def calculate_classification_evaluation(
    labels: Tensor,
    logits: Tensor,
    filepaths: Sequence[str],
    class_mapping: Mapping[str, int],
) -> ClassificationEvaluation:
    """Calculate all required classification metrics from aligned raw logits."""
    flat_labels = labels.detach().cpu().flatten()
    flat_logits = logits.detach().cpu().flatten()
    if flat_labels.numel() != flat_logits.numel() or flat_labels.numel() != len(
        filepaths
    ):
        raise ValueError("Labels, logits, and filepaths must have equal lengths")
    if set(flat_labels.int().tolist()) != {0, 1}:
        raise ValueError("Evaluation requires examples from both binary classes")
    ordered_names = _ordered_class_names(class_mapping)
    probabilities = torch.sigmoid(flat_logits)
    predictions = (probabilities >= 0.5).int()
    integer_labels = flat_labels.int()
    metrics = binary_classification_metrics(flat_labels, flat_logits)
    report = classification_report(
        integer_labels.numpy(),
        predictions.numpy(),
        labels=[0, 1],
        target_names=ordered_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        integer_labels.numpy(), predictions.numpy(), labels=[0, 1]
    )
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        integer_labels.numpy(), probabilities.numpy()
    )
    return ClassificationEvaluation(
        metrics=metrics,
        classification_report=report,
        confusion_matrix=matrix.astype(int).tolist(),
        false_positive_rate=false_positive_rate.tolist(),
        true_positive_rate=true_positive_rate.tolist(),
        thresholds=thresholds.tolist(),
        filepaths=list(filepaths),
        labels=integer_labels.tolist(),
        probabilities=probabilities.tolist(),
        predictions=predictions.tolist(),
    )


def load_best_classifier(
    checkpoint_path: str | Path,
    *,
    expected_prefix: str,
    device: torch.device | str = "cpu",
    classifier_factory: ClassifierFactory | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a best-model checkpoint directly into an evaluation-only classifier."""
    path = Path(checkpoint_path).expanduser()
    expected_name = f"{expected_prefix}_best.pt"
    if path.name != expected_name:
        raise ValueError(f"Expected best checkpoint named {expected_name}: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Best classifier checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Unable to load classifier checkpoint: {path}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Classifier checkpoint must contain a mapping")
    required = {
        "epoch",
        "model_state",
        "class_mapping",
        "configuration",
        "best_validation_f1",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Classifier checkpoint is missing keys: {sorted(missing)}")
    factory = classifier_factory or (
        lambda: create_binary_mobilenet(pretrained=False)
    )
    model = factory()
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "Classifier checkpoint has incompatible model state"
        ) from error
    model.to(torch.device(device)).eval()
    return model, dict(checkpoint)


def evaluate_model_comparison(
    *,
    test_manifest_path: str | Path,
    raw_directory: str | Path,
    classifier_a_checkpoint: str | Path,
    classifier_b_checkpoint: str | Path,
    generator_checkpoint: str | Path,
    metrics_directory: str | Path = "reports/metrics",
    figures_directory: str | Path = "reports/figures",
    batch_size: int = 32,
    num_workers: int = 2,
    device: torch.device | str = "cpu",
    include_model_b_originals: bool = True,
    classifier_factory: ClassifierFactory | None = None,
    generator_factory: GeneratorFactory | None = None,
) -> EvaluationArtifacts:
    """Evaluate both best classifiers on one aligned reserved-test stream."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    selected_device = torch.device(device)
    dataset = TestRecordDataset(test_manifest_path, raw_directory)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=selected_device.type == "cuda",
    )
    model_a, checkpoint_a = load_best_classifier(
        classifier_a_checkpoint,
        expected_prefix="classifier_a",
        device=selected_device,
        classifier_factory=classifier_factory,
    )
    model_b, checkpoint_b = load_best_classifier(
        classifier_b_checkpoint,
        expected_prefix="classifier_b",
        device=selected_device,
        classifier_factory=classifier_factory,
    )
    generator, generator_info = load_generator_checkpoint(
        generator_checkpoint,
        device=selected_device,
        generator_factory=generator_factory,
    )
    class_mapping = _validate_checkpoint_alignment(
        dataset,
        checkpoint_a,
        checkpoint_b,
        generator_info.identifier,
        generator_info.class_mapping,
    )

    filepaths: list[str] = []
    labels: list[Tensor] = []
    logits_a: list[Tensor] = []
    logits_b: list[Tensor] = []
    logits_b_original: list[Tensor] = []
    weighted_psnr = 0.0
    weighted_ssim = 0.0
    example_count = 0

    model_a.eval()
    model_b.eval()
    generator.eval()
    with torch.inference_mode():
        for original, low_resolution, high_resolution, batch_labels, paths in loader:
            original = original.to(selected_device, non_blocking=True)
            low_resolution = low_resolution.to(selected_device, non_blocking=True)
            high_resolution = high_resolution.to(selected_device, non_blocking=True)
            generated = generator(low_resolution)
            _validate_generated_test_batch(generated, len(batch_labels))
            generated_classifier_input = _imagenet_normalize_generated(generated)

            logits_a.append(model_a(original).flatten().cpu())
            logits_b.append(model_b(generated_classifier_input).flatten().cpu())
            if include_model_b_originals:
                logits_b_original.append(model_b(original).flatten().cpu())
            labels.append(batch_labels.cpu())
            filepaths.extend(paths)
            batch_count = len(batch_labels)
            weighted_psnr += (
                peak_signal_to_noise_ratio(generated, high_resolution) * batch_count
            )
            weighted_ssim += (
                structural_similarity(generated, high_resolution) * batch_count
            )
            example_count += batch_count

    all_labels = torch.cat(labels)
    evaluations = {
        "model_a_original": calculate_classification_evaluation(
            all_labels, torch.cat(logits_a), filepaths, class_mapping
        ),
        "model_b_srgan": calculate_classification_evaluation(
            all_labels, torch.cat(logits_b), filepaths, class_mapping
        ),
    }
    if include_model_b_originals:
        evaluations["model_b_original_secondary"] = (
            calculate_classification_evaluation(
                all_labels,
                torch.cat(logits_b_original),
                filepaths,
                class_mapping,
            )
        )
    psnr = weighted_psnr / example_count
    ssim = weighted_ssim / example_count
    return _save_evaluation_artifacts(
        evaluations=evaluations,
        class_mapping=class_mapping,
        psnr=psnr,
        ssim=ssim,
        metrics_directory=Path(metrics_directory).expanduser(),
        figures_directory=Path(figures_directory).expanduser(),
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        generator_identifier=generator_info.identifier,
    )


def _validate_checkpoint_alignment(
    dataset: TestRecordDataset,
    checkpoint_a: Mapping[str, Any],
    checkpoint_b: Mapping[str, Any],
    generator_identifier: str,
    generator_class_mapping: Mapping[str, int],
) -> dict[str, int]:
    manifest_mapping = {
        str(row.class_name): int(row.label)
        for row in dataset.manifest[["class_name", "label"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    mapping_a = _normalized_mapping(checkpoint_a["class_mapping"], "Model A")
    mapping_b = _normalized_mapping(checkpoint_b["class_mapping"], "Model B")
    mapping_generator = _normalized_mapping(generator_class_mapping, "generator")
    if (
        mapping_a != manifest_mapping
        or mapping_b != manifest_mapping
        or mapping_generator != manifest_mapping
    ):
        raise ValueError("Checkpoint class mappings do not match reserved test records")
    configuration_b = checkpoint_b["configuration"]
    if not isinstance(configuration_b, Mapping):
        raise ValueError("Model B checkpoint configuration must be a mapping")
    provenance = configuration_b.get("generator_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Model B checkpoint does not record generator provenance")
    if provenance.get("checkpoint_identifier") != generator_identifier:
        raise ValueError(
            "Model B was trained with a different generator checkpoint than evaluation"
        )
    return manifest_mapping


def _imagenet_normalize_generated(generated: Tensor) -> Tensor:
    display = ((generated + 1.0) / 2.0).clamp(0.0, 1.0)
    mean = display.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = display.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (display - mean) / std


def _validate_generated_test_batch(images: Tensor, expected_count: int) -> None:
    if images.shape != (expected_count, 3, 128, 128):
        raise ValueError(
            "Generator must return aligned RGB test tensors shaped "
            f"(N,3,128,128); received {tuple(images.shape)}"
        )
    if not torch.isfinite(images).all():
        raise ValueError("Generator returned non-finite test pixels")
    if images.min().item() < -1.00001 or images.max().item() > 1.00001:
        raise ValueError("Generator test output must remain in [-1, 1]")


def _ordered_class_names(class_mapping: Mapping[str, int]) -> list[str]:
    normalized = {str(name): int(label) for name, label in class_mapping.items()}
    if len(normalized) != 2 or set(normalized.values()) != {0, 1}:
        raise ValueError("Class mapping must map exactly two names to labels 0 and 1")
    return [name for name, _ in sorted(normalized.items(), key=lambda item: item[1])]


def _normalized_mapping(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} class mapping must be a mapping")
    return {str(key): int(label) for key, label in value.items()}


def _save_evaluation_artifacts(
    *,
    evaluations: Mapping[str, ClassificationEvaluation],
    class_mapping: Mapping[str, int],
    psnr: float,
    ssim: float,
    metrics_directory: Path,
    figures_directory: Path,
    checkpoint_a: Mapping[str, Any],
    checkpoint_b: Mapping[str, Any],
    generator_identifier: str,
) -> EvaluationArtifacts:
    metrics_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        [
            {"evaluation": name, **result.metrics}
            for name, result in evaluations.items()
        ]
    )
    comparison_path = metrics_directory / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    first = next(iter(evaluations.values()))
    prediction_rows: list[dict[str, Any]] = []
    for index, filepath in enumerate(first.filepaths):
        row: dict[str, Any] = {"filepath": filepath, "label": first.labels[index]}
        for name, result in evaluations.items():
            if result.filepaths != first.filepaths or result.labels != first.labels:
                raise RuntimeError("Classifier evaluation records are not aligned")
            row[f"{name}_probability"] = result.probabilities[index]
            row[f"{name}_prediction"] = result.predictions[index]
        prediction_rows.append(row)
    predictions_path = metrics_directory / "test_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    payload = {
        "record_count": len(first.filepaths),
        "class_mapping": dict(class_mapping),
        "checkpoints": {
            "model_a_epoch": int(checkpoint_a["epoch"]),
            "model_b_epoch": int(checkpoint_b["epoch"]),
            "generator_identifier": generator_identifier,
        },
        "srgan_image_quality": {"psnr_db": psnr, "ssim": ssim},
        "evaluations": {
            name: {
                "metrics": result.metrics,
                "classification_report": result.classification_report,
                "confusion_matrix": result.confusion_matrix,
                "roc_curve": {
                    "false_positive_rate": result.false_positive_rate,
                    "true_positive_rate": result.true_positive_rate,
                    "thresholds": result.thresholds,
                },
            }
            for name, result in evaluations.items()
        },
    }
    metrics_path = metrics_directory / "model_comparison.json"
    metrics_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    confusion_path = figures_directory / "confusion_matrices.png"
    roc_path = figures_directory / "roc_curves.png"
    _save_confusion_matrices(evaluations, class_mapping, confusion_path)
    _save_roc_curves(evaluations, roc_path)
    return EvaluationArtifacts(
        comparison_table=comparison,
        evaluations=dict(evaluations),
        psnr=psnr,
        ssim=ssim,
        metrics_json=metrics_path,
        comparison_csv=comparison_path,
        predictions_csv=predictions_path,
        confusion_figure=confusion_path,
        roc_figure=roc_path,
    )


def _save_confusion_matrices(
    evaluations: Mapping[str, ClassificationEvaluation],
    class_mapping: Mapping[str, int],
    path: Path,
) -> None:
    names = _ordered_class_names(class_mapping)
    figure, axes = plt.subplots(1, len(evaluations), figsize=(5 * len(evaluations), 4))
    axes_list = [axes] if len(evaluations) == 1 else list(axes)
    for axis, (title, result) in zip(axes_list, evaluations.items(), strict=True):
        matrix = result.confusion_matrix
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row][column]),
                    ha="center",
                    va="center",
                )
        axis.set_xticks([0, 1], names, rotation=20)
        axis.set_yticks([0, 1], names)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(title.replace("_", " ").title())
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_roc_curves(
    evaluations: Mapping[str, ClassificationEvaluation], path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, result in evaluations.items():
        auc = result.metrics["roc_auc"]
        axis.plot(
            result.false_positive_rate,
            result.true_positive_rate,
            label=f"{name.replace('_', ' ').title()} (AUC={auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Reserved-test ROC curves")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
