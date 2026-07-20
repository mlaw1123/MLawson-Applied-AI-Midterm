"""Leakage-safe classifier training, metrics, and resumable checkpoints."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

MetricValues = dict[str, float]
TrainingHistory = dict[str, list[MetricValues]]


def select_device() -> torch.device:
    """Select CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(random_seed: int) -> None:
    """Seed Python, NumPy, and every available PyTorch backend."""
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def binary_classification_metrics(labels: Tensor, logits: Tensor) -> MetricValues:
    """Convert logits to probabilities and calculate binary metrics."""
    labels_array = labels.detach().cpu().numpy().astype(int)
    probabilities = torch.sigmoid(logits.detach()).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "precision": float(
            precision_score(labels_array, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels_array, probabilities))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def run_classifier_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: Optimizer | None = None,
) -> MetricValues:
    """Run one training or evaluation epoch and return aggregate metrics."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    all_logits: list[Tensor] = []
    all_labels: list[Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images).flatten()
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if total_examples == 0:
        raise ValueError("DataLoader produced no examples")
    metrics = binary_classification_metrics(
        torch.cat(all_labels),
        torch.cat(all_logits),
    )
    metrics["loss"] = total_loss / total_examples
    return metrics


def save_classifier_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    history: TrainingHistory,
    class_mapping: Mapping[str, int],
    random_seed: int,
    configuration: Mapping[str, Any],
    best_validation_f1: float,
    data_loader_state: Tensor | None = None,
) -> None:
    """Atomically save all state needed to resume classifier training."""
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "history": deepcopy(history),
        "class_mapping": dict(class_mapping),
        "random_seed": random_seed,
        "configuration": dict(configuration),
        "best_validation_f1": best_validation_f1,
        "data_loader_state": data_loader_state,
        "rng_state": _capture_rng_state(),
    }
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_classifier_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    device: torch.device | str = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore classifier state and return its training metadata."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        # Preserve CPU RNG and DataLoader generator byte tensors. Model and
        # optimizer state is copied to the device of the positioned model.
        map_location="cpu",
        weights_only=False,
    )
    required = {
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "history",
        "class_mapping",
        "random_seed",
        "configuration",
        "best_validation_f1",
        "data_loader_state",
        "rng_state",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"Classifier checkpoint is missing keys: {', '.join(sorted(missing))}"
        )
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        if checkpoint["scheduler_state"] is None:
            raise ValueError("Checkpoint does not contain scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if restore_rng:
        _restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    validation_loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    epochs: int,
    device: torch.device,
    checkpoint_dir: str | Path,
    class_mapping: Mapping[str, int],
    random_seed: int,
    configuration: Mapping[str, Any],
    scheduler: LRScheduler | None = None,
    resume_from: str | Path | None = None,
    checkpoint_prefix: str = "classifier_a",
) -> TrainingHistory:
    """Train a classifier, saving separate latest and best-F1 checkpoints."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if re.fullmatch(r"[a-z0-9_]+", checkpoint_prefix) is None:
        raise ValueError("checkpoint_prefix must use lowercase letters, digits, or _")
    seed_everything(random_seed)
    model.to(device)
    destination = Path(checkpoint_dir).expanduser()
    latest_path = destination / f"{checkpoint_prefix}_latest.pt"
    best_path = destination / f"{checkpoint_prefix}_best.pt"
    history: TrainingHistory = {"train": [], "validation": []}
    start_epoch = 1
    best_validation_f1 = float("-inf")

    if resume_from is not None:
        resume_path = Path(resume_from).expanduser()
        if not resume_path.name.startswith(f"{checkpoint_prefix}_"):
            raise ValueError(
                "Resume checkpoint does not belong to the requested classifier: "
                f"expected prefix {checkpoint_prefix}, got {resume_path.name}"
            )
        checkpoint = load_classifier_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        if checkpoint["class_mapping"] != dict(class_mapping):
            raise ValueError("Checkpoint class mapping does not match current data")
        if checkpoint["random_seed"] != random_seed:
            raise ValueError("Checkpoint random seed does not match current run")
        if checkpoint["configuration"] != dict(configuration):
            raise ValueError(
                "Checkpoint configuration does not match current run or data provenance"
            )
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_f1 = float(checkpoint["best_validation_f1"])
        if (
            train_loader.generator is not None
            and checkpoint["data_loader_state"] is not None
        ):
            train_loader.generator.set_state(checkpoint["data_loader_state"])

    for epoch in range(start_epoch, epochs + 1):
        train_metrics = run_classifier_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        validation_metrics = run_classifier_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        train_metrics["epoch"] = float(epoch)
        validation_metrics["epoch"] = float(epoch)
        history["train"].append(train_metrics)
        history["validation"].append(validation_metrics)
        if scheduler is not None:
            scheduler.step()

        improved = validation_metrics["f1"] > best_validation_f1
        if improved:
            best_validation_f1 = validation_metrics["f1"]
        checkpoint_arguments = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "history": history,
            "class_mapping": class_mapping,
            "random_seed": random_seed,
            "configuration": configuration,
            "best_validation_f1": best_validation_f1,
            "data_loader_state": (
                train_loader.generator.get_state()
                if train_loader.generator is not None
                else None
            ),
        }
        save_classifier_checkpoint(latest_path, **checkpoint_arguments)
        if improved:
            save_classifier_checkpoint(best_path, **checkpoint_arguments)

    return history


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
