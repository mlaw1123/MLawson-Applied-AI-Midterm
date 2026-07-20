"""Synthetic tests for resumable SRGAN image generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as torch_functional

from applied_ai_midterm.data import MANIFEST_COLUMNS
from applied_ai_midterm.generation import (
    GENERATED_MANIFEST_COLUMNS,
    PROVENANCE_FILENAME,
    generate_model_b_dataset,
    load_generator_checkpoint,
)


class TinyGenerator(nn.Module):
    """Parameter-free 4x generator used to avoid real checkpoints in tests."""

    calls = 0

    def forward(self, inputs: Tensor) -> Tensor:
        type(self).calls += 1
        return torch_functional.interpolate(inputs, scale_factor=4, mode="nearest")


def tiny_generator_factory(configuration: dict[str, Any]) -> nn.Module:
    """Build the test generator while accepting checkpoint configuration."""
    assert configuration["low_resolution_size"] == 32
    return TinyGenerator()


def build_generation_inputs(root: Path) -> tuple[Path, Path, Path]:
    """Create nested duplicate names, a train manifest, and a fake checkpoint."""
    raw_directory = root / "raw"
    records: list[dict[str, str | int]] = []
    for label, class_name in enumerate(("alpha", "zeta")):
        for folder in ("one", "two"):
            relative = Path(class_name) / folder / "same.png"
            destination = raw_directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB", (50, 45), color=(label * 180, len(folder) * 20, 90)
            ).save(destination)
            records.append(
                {
                    "filepath": relative.as_posix(),
                    "class_name": class_name,
                    "label": label,
                }
            )
    manifest_path = root / "splits" / "train.csv"
    manifest_path.parent.mkdir(parents=True)
    pd.DataFrame(records, columns=MANIFEST_COLUMNS).to_csv(
        manifest_path, index=False
    )
    checkpoint_path = root / "srgan_epoch_0150.pt"
    save_fake_checkpoint(checkpoint_path)
    return raw_directory, manifest_path, checkpoint_path


def save_fake_checkpoint(path: Path, *, marker: str = "first") -> None:
    """Write a completed checkpoint compatible with ``TinyGenerator``."""
    torch.save(
        {
            "epoch": 150,
            "generator_state": TinyGenerator().state_dict(),
            "configuration": {
                "low_resolution_size": 32,
                "high_resolution_size": 128,
                "srgan_epochs": 150,
                "residual_blocks": 1,
                "test_marker": marker,
            },
            "class_mapping": {"alpha": 0, "zeta": 1},
        },
        path,
    )


def generation_kwargs(root: Path) -> dict[str, object]:
    """Return common test arguments using only temporary paths."""
    raw_directory, manifest_path, checkpoint_path = build_generation_inputs(root)
    return {
        "checkpoint_path": checkpoint_path,
        "raw_directory": raw_directory,
        "train_manifest_path": manifest_path,
        "output_directory": root / "generated" / "model_b_train",
        "output_manifest_path": root / "generated" / "model_b_train.csv",
        "batch_size": 2,
        "show_progress": False,
        "generator_factory": tiny_generator_factory,
    }


def test_generation_preserves_labels_dimensions_and_unique_names(
    tmp_path: Path,
) -> None:
    kwargs = generation_kwargs(tmp_path)
    summary = generate_model_b_dataset(**kwargs)
    manifest = pd.read_csv(kwargs["output_manifest_path"])

    assert summary.total == 4
    assert summary.generated == 4
    assert summary.skipped == 0
    assert list(manifest.columns) == GENERATED_MANIFEST_COLUMNS
    assert manifest["filepath"].is_unique
    assert manifest["source_filepath"].str.contains("same.png", regex=False).all()
    assert dict(
        manifest[["class_name", "label"]].drop_duplicates().values.tolist()
    ) == {"alpha": 0, "zeta": 1}
    for filepath in manifest["filepath"]:
        generated_path = Path(kwargs["output_manifest_path"]).parent / filepath
        with Image.open(generated_path) as image:
            image.load()
            assert image.mode == "RGB"
            assert image.size == (128, 128)
    assert (Path(kwargs["output_directory"]) / PROVENANCE_FILENAME).is_file()


def test_generation_resume_skips_valid_outputs_and_repairs_corruption(
    tmp_path: Path,
) -> None:
    kwargs = generation_kwargs(tmp_path)
    TinyGenerator.calls = 0
    generate_model_b_dataset(**kwargs)
    original_calls = TinyGenerator.calls

    resumed = generate_model_b_dataset(**kwargs)
    assert resumed.generated == 0
    assert resumed.skipped == 4
    assert TinyGenerator.calls == original_calls

    manifest = pd.read_csv(kwargs["output_manifest_path"])
    corrupt_path = Path(kwargs["output_manifest_path"]).parent / manifest.iloc[0][
        "filepath"
    ]
    corrupt_path.write_bytes(b"corrupt")
    repaired = generate_model_b_dataset(**kwargs)
    assert repaired.generated == 1
    assert repaired.skipped == 3
    with Image.open(corrupt_path) as image:
        assert image.size == (128, 128)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    kwargs = generation_kwargs(tmp_path)
    kwargs["dry_run"] = True

    summary = generate_model_b_dataset(**kwargs)

    assert summary.generated == 4
    assert summary.total == 4
    assert not Path(kwargs["output_directory"]).exists()
    assert not Path(kwargs["output_manifest_path"]).exists()


def test_different_checkpoint_cannot_overwrite_outputs(tmp_path: Path) -> None:
    kwargs = generation_kwargs(tmp_path)
    generate_model_b_dataset(**kwargs)
    second_checkpoint = tmp_path / "srgan_epoch_0150_second.pt"
    save_fake_checkpoint(second_checkpoint, marker="second")
    kwargs["checkpoint_path"] = second_checkpoint

    with pytest.raises(ValueError, match="different checkpoint or configuration"):
        generate_model_b_dataset(**kwargs)


def test_test_manifest_is_rejected_before_generation(tmp_path: Path) -> None:
    kwargs = generation_kwargs(tmp_path)
    train_path = Path(kwargs["train_manifest_path"])
    test_path = train_path.with_name("test.csv")
    train_path.replace(test_path)
    kwargs["train_manifest_path"] = test_path

    with pytest.raises(ValueError, match="train.csv only"):
        generate_model_b_dataset(**kwargs)


def test_incompatible_or_incomplete_checkpoint_is_rejected(tmp_path: Path) -> None:
    kwargs = generation_kwargs(tmp_path)
    checkpoint_path = Path(kwargs["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["configuration"]["high_resolution_size"] = 64
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="32x32 inputs and 128x128 outputs"):
        load_generator_checkpoint(
            checkpoint_path, generator_factory=tiny_generator_factory
        )
