"""Synthetic tests for SRGAN models, data, losses, metrics, and checkpoints."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import applied_ai_midterm.srgan as srgan_module
from applied_ai_midterm.data import MANIFEST_COLUMNS
from applied_ai_midterm.srgan import Discriminator, Generator, GeneratorLoss
from applied_ai_midterm.srgan_data import (
    SRGANManifestDataset,
    create_srgan_dataloaders,
)
from applied_ai_midterm.srgan_training import (
    find_latest_valid_checkpoint,
    load_srgan_checkpoint,
    peak_signal_to_noise_ratio,
    save_srgan_checkpoint,
    structural_similarity,
    train_srgan_step,
)
from applied_ai_midterm.transforms import SRGANPairTransform


def build_training_manifest(
    root: Path,
    images_per_class: int = 10,
) -> tuple[Path, Path]:
    """Create a temporary binary train manifest with generated RGB images."""
    raw_dir = root / "raw"
    records = []
    for label, class_name in enumerate(("alpha", "zeta")):
        for index in range(images_per_class):
            relative_path = Path(class_name) / f"image_{index}.png"
            image_path = raw_dir / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (48, 48),
                color=(label * 180, index * 8, 90),
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


def test_generator_and_discriminator_shapes_and_ranges() -> None:
    generator = Generator(residual_blocks=2, channels=16)
    discriminator = Discriminator(base_channels=8)
    low_resolution = torch.randn(2, 3, 32, 32)

    generated = generator(low_resolution)
    logits = discriminator(generated)

    assert generated.shape == (2, 3, 128, 128)
    assert torch.all((generated >= -1) & (generated <= 1))
    assert logits.shape == (2,)
    assert not any(isinstance(module, nn.Sigmoid) for module in discriminator.modules())


def test_generator_loss_components_and_gradients() -> None:
    generated = torch.zeros(2, 3, 16, 16, requires_grad=True)
    target = torch.ones_like(generated)
    discriminator_logits = torch.tensor([-0.5, 0.5], requires_grad=True)
    criterion = GeneratorLoss(
        pixel_weight=1.0,
        adversarial_weight=0.1,
        perceptual_weight=0.2,
        perceptual_extractor=nn.Identity(),
    )

    components = criterion(generated, target, discriminator_logits)
    components["total"].backward()

    assert set(components) == {"total", "pixel", "adversarial", "perceptual"}
    assert components["pixel"].item() == pytest.approx(1.0)
    assert components["perceptual"].item() == pytest.approx(1.0)
    assert generated.grad is not None
    assert discriminator_logits.grad is not None


def test_disabled_perceptual_loss_never_constructs_vgg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> nn.Module:
        del args, kwargs
        raise AssertionError("VGG must not be created when perceptual loss is disabled")

    monkeypatch.setattr(srgan_module, "VGGFeatureExtractor", fail_if_called)
    criterion = GeneratorLoss(perceptual_weight=0.0)
    assert criterion.perceptual_extractor is None


def test_srgan_dataloaders_use_only_training_manifest(tmp_path: Path) -> None:
    manifest_path, raw_dir = build_training_manifest(tmp_path)

    loaders = create_srgan_dataloaders(
        manifest_path,
        raw_dir,
        batch_size=4,
        validation_ratio=0.20,
        random_seed=42,
        num_workers=0,
        fixed_sample_count=3,
    )
    low_resolution, high_resolution = next(iter(loaders.train))

    assert loaders.train_size == 16
    assert loaders.validation_size == 4
    assert loaders.class_mapping == {"alpha": 0, "zeta": 1}
    assert low_resolution.shape == (4, 3, 32, 32)
    assert high_resolution.shape == (4, 3, 128, 128)
    assert loaders.fixed_low_resolution.shape == (3, 3, 32, 32)
    assert loaders.fixed_high_resolution.shape == (3, 3, 128, 128)


def test_srgan_dataset_reports_corrupt_image(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    corrupt_path = raw_dir / "alpha" / "broken.png"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"not an image")
    manifest = pd.DataFrame(
        [["alpha/broken.png", "alpha", 0]],
        columns=MANIFEST_COLUMNS,
    )
    dataset = SRGANManifestDataset(
        manifest,
        raw_dir,
        SRGANPairTransform(32, 128, training=False),
    )

    with pytest.raises(RuntimeError, match="Unable to read SRGAN image"):
        dataset[0]


def test_image_quality_metrics_for_identical_images() -> None:
    images = torch.rand(2, 3, 32, 32) * 2 - 1

    assert peak_signal_to_noise_ratio(images, images) == float("inf")
    assert structural_similarity(images, images) == pytest.approx(1.0, abs=1e-5)


def test_single_real_srgan_training_step() -> None:
    generator = Generator(residual_blocks=1, channels=8)
    discriminator = Discriminator(base_channels=8)
    generator_optimizer = Adam(generator.parameters(), lr=1e-4)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=1e-4)
    low_resolution = torch.randn(2, 3, 16, 16)
    high_resolution = torch.randn(2, 3, 64, 64).clamp(-1, 1)

    metrics = train_srgan_step(
        generator,
        discriminator,
        low_resolution,
        high_resolution,
        generator_optimizer,
        discriminator_optimizer,
        GeneratorLoss(perceptual_weight=0.0),
        torch.device("cpu"),
    )

    assert set(metrics) == {
        "discriminator_loss",
        "generator_loss",
        "pixel_loss",
        "adversarial_loss",
        "perceptual_loss",
    }
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_srgan_checkpoint_round_trip_and_latest_valid_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = Generator(residual_blocks=1, channels=8)
    discriminator = Discriminator(base_channels=8)
    generator_optimizer = Adam(generator.parameters(), lr=1e-4)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=1e-4)
    generator_scheduler = StepLR(generator_optimizer, step_size=1)
    discriminator_scheduler = StepLR(discriminator_optimizer, step_size=1)
    checkpoint_path = tmp_path / "srgan_epoch_0005.pt"
    original_parameter = next(generator.parameters()).detach().clone()

    save_srgan_checkpoint(
        checkpoint_path,
        epoch=5,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        history=[{"epoch": 5.0, "generator_loss": 1.0}],
        configuration={"epochs": 150},
        random_seed=42,
        class_mapping={"alpha": 0, "zeta": 1},
        fixed_low_resolution=torch.zeros(2, 3, 32, 32),
        fixed_high_resolution=torch.zeros(2, 3, 128, 128),
        data_loader_state=torch.Generator().manual_seed(42).get_state(),
    )
    with torch.no_grad():
        next(generator.parameters()).zero_()
    original_load = torch.load
    map_locations: list[object] = []

    def recording_load(*args: object, **kwargs: object) -> object:
        map_locations.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    checkpoint = load_srgan_checkpoint(
        checkpoint_path,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
    )

    assert torch.equal(next(generator.parameters()), original_parameter)
    assert checkpoint["epoch"] == 5
    assert checkpoint["generator_scheduler_state"] is not None
    assert checkpoint["discriminator_scheduler_state"] is not None
    assert checkpoint["fixed_validation"]["low_resolution"].shape[0] == 2
    assert map_locations[-1] == "cpu"
    assert find_latest_valid_checkpoint(tmp_path) == checkpoint_path

    corrupt_path = tmp_path / "srgan_epoch_0010.pt"
    corrupt_path.write_bytes(b"incomplete checkpoint")
    assert find_latest_valid_checkpoint(tmp_path) == checkpoint_path
