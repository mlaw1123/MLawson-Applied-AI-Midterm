"""SRGAN optimization, image-quality metrics, and resumable checkpoints."""

from __future__ import annotations

import pickle
import random
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from applied_ai_midterm.srgan import GeneratorLoss
from applied_ai_midterm.training import seed_everything

SRGANHistory = list[dict[str, float]]


def peak_signal_to_noise_ratio(generated: Tensor, target: Tensor) -> float:
    """Calculate mean PSNR in dB for SRGAN tensors in ``[-1, 1]``."""
    generated_display = ((generated.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
    target_display = ((target.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
    mse = (generated_display - target_display).square().flatten(1).mean(dim=1)
    psnr = torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(1.0 / mse),
    )
    return float(psnr.mean().cpu())


def structural_similarity(generated: Tensor, target: Tensor) -> float:
    """Calculate windowed SSIM for RGB SRGAN tensors in ``[-1, 1]``."""
    generated_display = ((generated.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
    target_display = ((target.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
    if generated_display.shape != target_display.shape or generated_display.ndim != 4:
        raise ValueError("SSIM expects matching tensors with shape (N,C,H,W)")
    window = min(11, generated_display.shape[-2], generated_display.shape[-1])
    if window % 2 == 0:
        window -= 1
    if window < 1:
        raise ValueError("SSIM images must have non-empty spatial dimensions")
    padding = window // 2
    mean_generated = F.avg_pool2d(
        generated_display,
        window,
        stride=1,
        padding=padding,
    )
    mean_target = F.avg_pool2d(
        target_display,
        window,
        stride=1,
        padding=padding,
    )
    variance_generated = F.avg_pool2d(
        generated_display.square(),
        window,
        stride=1,
        padding=padding,
    ) - mean_generated.square()
    variance_target = F.avg_pool2d(
        target_display.square(),
        window,
        stride=1,
        padding=padding,
    ) - mean_target.square()
    covariance = F.avg_pool2d(
        generated_display * target_display,
        window,
        stride=1,
        padding=padding,
    ) - mean_generated * mean_target
    constant_1 = 0.01**2
    constant_2 = 0.03**2
    numerator = (2 * mean_generated * mean_target + constant_1) * (
        2 * covariance + constant_2
    )
    denominator = (
        mean_generated.square() + mean_target.square() + constant_1
    ) * (variance_generated + variance_target + constant_2)
    return float((numerator / denominator.clamp_min(1e-12)).mean().cpu())


def train_srgan_step(
    generator: nn.Module,
    discriminator: nn.Module,
    low_resolution: Tensor,
    high_resolution: Tensor,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    generator_loss: GeneratorLoss,
    device: torch.device,
) -> dict[str, float]:
    """Perform one alternating discriminator and generator optimization step."""
    generator.train()
    discriminator.train()
    low_resolution = low_resolution.to(device, non_blocking=True)
    high_resolution = high_resolution.to(device, non_blocking=True)
    generated = generator(low_resolution)

    discriminator_optimizer.zero_grad(set_to_none=True)
    real_logits = discriminator(high_resolution)
    generated_logits = discriminator(generated.detach())
    discriminator_criterion = nn.BCEWithLogitsLoss()
    real_loss = discriminator_criterion(real_logits, torch.ones_like(real_logits))
    generated_loss = discriminator_criterion(
        generated_logits,
        torch.zeros_like(generated_logits),
    )
    discriminator_loss = 0.5 * (real_loss + generated_loss)
    discriminator_loss.backward()
    discriminator_optimizer.step()

    discriminator.eval()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    generator_optimizer.zero_grad(set_to_none=True)
    generator_logits = discriminator(generated)
    components = generator_loss(generated, high_resolution, generator_logits)
    components["total"].backward()
    generator_optimizer.step()
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    discriminator.train()

    return {
        "discriminator_loss": float(discriminator_loss.detach()),
        "generator_loss": float(components["total"].detach()),
        "pixel_loss": float(components["pixel"].detach()),
        "adversarial_loss": float(components["adversarial"].detach()),
        "perceptual_loss": float(components["perceptual"].detach()),
    }


def run_srgan_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    generator_loss: GeneratorLoss,
    device: torch.device,
) -> dict[str, float]:
    """Aggregate named SRGAN losses across one training epoch."""
    totals: dict[str, float] = {}
    example_count = 0
    for low_resolution, high_resolution in loader:
        batch_metrics = train_srgan_step(
            generator,
            discriminator,
            low_resolution,
            high_resolution,
            generator_optimizer,
            discriminator_optimizer,
            generator_loss,
            device,
        )
        batch_size = low_resolution.shape[0]
        example_count += batch_size
        for name, value in batch_metrics.items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
    if example_count == 0:
        raise ValueError("SRGAN DataLoader produced no training examples")
    return {name: value / example_count for name, value in totals.items()}


def evaluate_srgan(
    generator: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate mean PSNR and SSIM without updating the generator."""
    generator.eval()
    weighted_psnr = 0.0
    weighted_ssim = 0.0
    example_count = 0
    with torch.no_grad():
        for low_resolution, high_resolution in loader:
            low_resolution = low_resolution.to(device, non_blocking=True)
            high_resolution = high_resolution.to(device, non_blocking=True)
            generated = generator(low_resolution)
            batch_size = low_resolution.shape[0]
            weighted_psnr += (
                peak_signal_to_noise_ratio(generated, high_resolution) * batch_size
            )
            weighted_ssim += (
                structural_similarity(generated, high_resolution) * batch_size
            )
            example_count += batch_size
    if example_count == 0:
        raise ValueError("SRGAN validation DataLoader produced no examples")
    return {
        "validation_psnr": weighted_psnr / example_count,
        "validation_ssim": weighted_ssim / example_count,
    }


def save_srgan_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    generator_scheduler: LRScheduler | None,
    discriminator_scheduler: LRScheduler | None,
    history: SRGANHistory,
    configuration: Mapping[str, Any],
    random_seed: int,
    class_mapping: Mapping[str, int],
    fixed_low_resolution: Tensor,
    fixed_high_resolution: Tensor,
    data_loader_state: Tensor | None = None,
) -> None:
    """Atomically save every state required to resume SRGAN training."""
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    checkpoint = {
        "epoch": epoch,
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "generator_optimizer_state": generator_optimizer.state_dict(),
        "discriminator_optimizer_state": discriminator_optimizer.state_dict(),
        "generator_scheduler_state": (
            generator_scheduler.state_dict()
            if generator_scheduler is not None
            else None
        ),
        "discriminator_scheduler_state": (
            discriminator_scheduler.state_dict()
            if discriminator_scheduler is not None
            else None
        ),
        "history": deepcopy(history),
        "configuration": dict(configuration),
        "random_seed": random_seed,
        "class_mapping": dict(class_mapping),
        "fixed_validation": {
            "low_resolution": fixed_low_resolution.detach().cpu(),
            "high_resolution": fixed_high_resolution.detach().cpu(),
        },
        "data_loader_state": data_loader_state,
        "rng_state": _capture_rng_state(),
    }
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_srgan_checkpoint(
    path: str | Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    generator_scheduler: LRScheduler | None = None,
    discriminator_scheduler: LRScheduler | None = None,
    device: torch.device | str = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore SRGAN model, optimizer, scheduler, history, and random state."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SRGAN checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path,
        # RNG and DataLoader generator states must remain CPU byte tensors.
        # Model/optimizer load_state_dict calls copy their state to the device
        # of the already-positioned parameters.
        map_location="cpu",
        weights_only=False,
    )
    missing = _required_checkpoint_keys() - checkpoint.keys()
    if missing:
        raise ValueError(f"SRGAN checkpoint is missing: {', '.join(sorted(missing))}")
    generator.load_state_dict(checkpoint["generator_state"])
    discriminator.load_state_dict(checkpoint["discriminator_state"])
    generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state"])
    discriminator_optimizer.load_state_dict(
        checkpoint["discriminator_optimizer_state"]
    )
    _restore_scheduler(
        generator_scheduler,
        checkpoint["generator_scheduler_state"],
        "generator",
    )
    _restore_scheduler(
        discriminator_scheduler,
        checkpoint["discriminator_scheduler_state"],
        "discriminator",
    )
    if restore_rng:
        _restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def find_latest_valid_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the newest readable complete checkpoint, skipping corrupt files."""
    destination = Path(checkpoint_dir).expanduser()
    candidates = sorted(destination.glob("srgan_epoch_*.pt"), reverse=True)
    required = _required_checkpoint_keys()
    for candidate in candidates:
        try:
            checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
        except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
            continue
        if isinstance(checkpoint, dict) and not (required - checkpoint.keys()):
            return candidate
    return None


def fit_srgan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    validation_loader: DataLoader[tuple[Tensor, Tensor]],
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    generator_loss: GeneratorLoss,
    *,
    epochs: int,
    checkpoint_interval: int,
    device: torch.device,
    checkpoint_dir: str | Path,
    class_mapping: Mapping[str, int],
    random_seed: int,
    configuration: Mapping[str, Any],
    fixed_low_resolution: Tensor,
    fixed_high_resolution: Tensor,
    generator_scheduler: LRScheduler | None = None,
    discriminator_scheduler: LRScheduler | None = None,
    automatic_resume: bool = True,
) -> SRGANHistory:
    """Train SRGAN for at least 150 epochs with five-epoch checkpoints."""
    if epochs < 150:
        raise ValueError("SRGAN training requires at least 150 epochs")
    if checkpoint_interval != 5:
        raise ValueError("SRGAN checkpoints must be saved every five epochs")
    seed_everything(random_seed)
    generator.to(device)
    discriminator.to(device)
    generator_loss.to(device)
    destination = Path(checkpoint_dir).expanduser()
    history: SRGANHistory = []
    start_epoch = 1

    resume_path = (
        find_latest_valid_checkpoint(destination) if automatic_resume else None
    )
    if resume_path is not None:
        checkpoint = load_srgan_checkpoint(
            resume_path,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator_scheduler=generator_scheduler,
            discriminator_scheduler=discriminator_scheduler,
            device=device,
        )
        if checkpoint["random_seed"] != random_seed:
            raise ValueError("SRGAN checkpoint random seed does not match current run")
        if checkpoint["class_mapping"] != dict(class_mapping):
            raise ValueError(
                "SRGAN checkpoint class mapping does not match current data"
            )
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"]) + 1
        fixed_low_resolution = checkpoint["fixed_validation"]["low_resolution"]
        fixed_high_resolution = checkpoint["fixed_validation"]["high_resolution"]
        if (
            train_loader.generator is not None
            and checkpoint["data_loader_state"] is not None
        ):
            train_loader.generator.set_state(checkpoint["data_loader_state"])

    for epoch in range(start_epoch, epochs + 1):
        metrics = run_srgan_epoch(
            generator,
            discriminator,
            train_loader,
            generator_optimizer,
            discriminator_optimizer,
            generator_loss,
            device,
        )
        metrics.update(evaluate_srgan(generator, validation_loader, device))
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        if generator_scheduler is not None:
            generator_scheduler.step()
        if discriminator_scheduler is not None:
            discriminator_scheduler.step()
        print(
            f"Epoch {epoch:03d}/{epochs}: G={metrics['generator_loss']:.4f}, "
            f"D={metrics['discriminator_loss']:.4f}, "
            f"PSNR={metrics['validation_psnr']:.2f}, "
            f"SSIM={metrics['validation_ssim']:.4f}"
        )

        if epoch % checkpoint_interval == 0:
            checkpoint_path = destination / f"srgan_epoch_{epoch:04d}.pt"
            save_srgan_checkpoint(
                checkpoint_path,
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                generator_scheduler=generator_scheduler,
                discriminator_scheduler=discriminator_scheduler,
                history=history,
                configuration=configuration,
                random_seed=random_seed,
                class_mapping=class_mapping,
                fixed_low_resolution=fixed_low_resolution,
                fixed_high_resolution=fixed_high_resolution,
                data_loader_state=(
                    train_loader.generator.get_state()
                    if train_loader.generator is not None
                    else None
                ),
            )
            _save_progress_samples(
                destination / "samples" / f"epoch_{epoch:04d}.pt",
                generator,
                fixed_low_resolution,
                fixed_high_resolution,
                device,
                epoch,
                metrics,
            )
    return history


def _save_progress_samples(
    path: Path,
    generator: nn.Module,
    low_resolution: Tensor,
    high_resolution: Tensor,
    device: torch.device,
    epoch: int,
    metrics: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        generated = generator(low_resolution.to(device)).cpu()
    torch.save(
        {
            "epoch": epoch,
            "low_resolution": low_resolution.cpu(),
            "generated": generated,
            "high_resolution": high_resolution.cpu(),
            "metrics": dict(metrics),
        },
        path,
    )


def _restore_scheduler(
    scheduler: LRScheduler | None,
    state: dict[str, Any] | None,
    name: str,
) -> None:
    if scheduler is not None:
        if state is None:
            raise ValueError(f"Checkpoint does not contain {name} scheduler state")
        scheduler.load_state_dict(state)


def _required_checkpoint_keys() -> set[str]:
    return {
        "epoch",
        "generator_state",
        "discriminator_state",
        "generator_optimizer_state",
        "discriminator_optimizer_state",
        "generator_scheduler_state",
        "discriminator_scheduler_state",
        "history",
        "configuration",
        "random_seed",
        "class_mapping",
        "fixed_validation",
        "data_loader_state",
        "rng_state",
    }


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
