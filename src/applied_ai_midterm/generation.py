"""Create the leakage-safe generated-image training set for Model B."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from tqdm.auto import tqdm

from applied_ai_midterm.data import load_split_manifest
from applied_ai_midterm.srgan import Generator
from applied_ai_midterm.transforms import SRGANPairTransform, denormalize_srgan

GENERATED_MANIFEST_COLUMNS = [
    "filepath",
    "source_filepath",
    "class_name",
    "label",
]
PROVENANCE_FILENAME = ".generation_metadata.json"
GeneratorFactory = Callable[[Mapping[str, Any]], nn.Module]


@dataclass(frozen=True)
class GeneratorCheckpoint:
    """Validated generator and the provenance needed to identify its outputs."""

    identifier: str
    path: Path
    epoch: int
    configuration: dict[str, Any]
    class_mapping: dict[str, int]


@dataclass(frozen=True)
class GenerationSummary:
    """Counts and destinations produced by one generation invocation."""

    total: int
    generated: int
    skipped: int
    dry_run: bool
    output_directory: Path
    manifest_path: Path
    checkpoint_identifier: str


class _GenerationDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        records: pd.DataFrame,
        pending_indices: list[int],
        raw_directory: Path,
        transform: SRGANPairTransform,
    ) -> None:
        self.records = records
        self.pending_indices = pending_indices
        self.raw_directory = raw_directory
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pending_indices)

    def __getitem__(self, item: int) -> tuple[Tensor, int]:
        index = self.pending_indices[item]
        source_path = self.raw_directory / str(self.records.iloc[index]["filepath"])
        try:
            with Image.open(source_path) as image:
                low_resolution, _ = self.transform(image)
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Unable to read source image: {source_path}") from error
        return low_resolution, index


def load_generator_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    generator_factory: GeneratorFactory | None = None,
) -> tuple[nn.Module, GeneratorCheckpoint]:
    """Load and strictly validate a completed 32-to-128 SRGAN checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SRGAN generator checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Unable to load SRGAN checkpoint: {path}") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("SRGAN checkpoint must contain a mapping")

    required = {"epoch", "generator_state", "configuration", "class_mapping"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(
            f"SRGAN checkpoint is missing required keys: {sorted(missing)}"
        )
    configuration = _validate_configuration(checkpoint["configuration"])
    class_mapping = _validate_class_mapping(checkpoint["class_mapping"])
    epoch = _positive_int(checkpoint["epoch"], "checkpoint epoch")
    planned_epochs = _positive_int(
        configuration.get("srgan_epochs", 150), "configuration srgan_epochs"
    )
    if planned_epochs < 150 or epoch < planned_epochs:
        raise ValueError(
            "Generator checkpoint is not a completed 150-epoch SRGAN run: "
            f"checkpoint epoch={epoch}, configured epochs={planned_epochs}"
        )

    factory = generator_factory or _default_generator_factory
    generator = factory(configuration)
    try:
        generator.load_state_dict(checkpoint["generator_state"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "Generator checkpoint parameters are incompatible with its configuration"
        ) from error
    selected_device = torch.device(device)
    generator.to(selected_device).eval()
    metadata = GeneratorCheckpoint(
        identifier=_sha256(path),
        path=path,
        epoch=epoch,
        configuration=configuration,
        class_mapping=class_mapping,
    )
    return generator, metadata


def generate_model_b_dataset(
    checkpoint_path: str | Path,
    *,
    raw_directory: str | Path = "data/raw",
    train_manifest_path: str | Path = "data/splits/train.csv",
    output_directory: str | Path = "data/generated/model_b_train",
    output_manifest_path: str | Path = "data/generated/model_b_train.csv",
    device: torch.device | str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
    dry_run: bool = False,
    show_progress: bool = True,
    generator_factory: GeneratorFactory | None = None,
) -> GenerationSummary:
    """Generate Model B training images from the persisted training split only.

    Existing readable 128×128 outputs are skipped when their provenance matches
    the supplied checkpoint. Outputs from an unknown or different checkpoint
    cause an error instead of being overwritten.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    manifest_path = Path(train_manifest_path).expanduser().resolve()
    if manifest_path.name != "train.csv":
        raise ValueError("Model B generation must read data/splits/train.csv only")
    raw_path = Path(raw_directory).expanduser().resolve()
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_path}")
    records = load_split_manifest(manifest_path)
    _validate_sources(records, raw_path)

    selected_device = torch.device(device)
    generator, checkpoint = load_generator_checkpoint(
        checkpoint_path,
        device=selected_device,
        generator_factory=generator_factory,
    )
    manifest_mapping = {
        str(row.class_name): int(row.label)
        for row in records[["class_name", "label"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if manifest_mapping != checkpoint.class_mapping:
        raise ValueError(
            "Checkpoint class mapping does not match the training manifest: "
            f"checkpoint={checkpoint.class_mapping}, manifest={manifest_mapping}"
        )

    output_path = Path(output_directory).expanduser().resolve()
    generated_manifest_path = Path(output_manifest_path).expanduser().resolve()
    metadata_path = output_path / PROVENANCE_FILENAME
    expected_metadata = _provenance(checkpoint, manifest_path)
    _validate_existing_provenance(output_path, metadata_path, expected_metadata)

    expected_paths = [
        _generated_path(output_path, str(row.filepath), str(row.class_name))
        for row in records.itertuples(index=False)
    ]
    completed = {
        index
        for index, path in enumerate(expected_paths)
        if path.is_file() and _is_valid_generated_image(path)
    }
    pending = [index for index in range(len(records)) if index not in completed]
    summary = GenerationSummary(
        total=len(records),
        generated=len(pending),
        skipped=len(completed),
        dry_run=dry_run,
        output_directory=output_path,
        manifest_path=generated_manifest_path,
        checkpoint_identifier=checkpoint.identifier,
    )
    if dry_run:
        return summary

    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(metadata_path, expected_metadata)
    completed_rows = {
        index: _manifest_record(
            records.iloc[index], expected_paths[index], generated_manifest_path.parent
        )
        for index in completed
    }
    _write_manifest(completed_rows, generated_manifest_path)

    transform = SRGANPairTransform(32, 128, training=False)
    dataset = _GenerationDataset(records, pending, raw_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=selected_device.type == "cuda",
    )
    progress = tqdm(
        loader,
        total=len(loader),
        desc="Generating Model B images",
        disable=not show_progress,
    )
    with torch.inference_mode():
        for low_resolution, indices in progress:
            generated = generator(low_resolution.to(selected_device))
            _validate_generated_batch(generated, len(indices))
            batch_pairs = zip(generated.cpu(), indices, strict=True)
            for image_tensor, index_tensor in batch_pairs:
                index = int(index_tensor.item())
                destination = expected_paths[index]
                _save_generated_image(image_tensor, destination)
                completed_rows[index] = _manifest_record(
                    records.iloc[index], destination, generated_manifest_path.parent
                )
            _write_manifest(completed_rows, generated_manifest_path)

    if len(completed_rows) != len(records):
        raise RuntimeError("Generation ended before every training record was saved")
    for destination in expected_paths:
        if not _is_valid_generated_image(destination):
            raise ValueError(f"Generated image failed final validation: {destination}")
    _write_manifest(completed_rows, generated_manifest_path)
    return summary


def _default_generator_factory(configuration: Mapping[str, Any]) -> nn.Module:
    return Generator(
        residual_blocks=_positive_int(
            configuration.get("residual_blocks", 16), "residual_blocks"
        ),
        channels=_positive_int(
            configuration.get("generator_channels", 64), "generator_channels"
        ),
    )


def _validate_configuration(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint configuration must be a mapping")
    configuration = dict(value)
    low_size = _positive_int(
        configuration.get("low_resolution_size"), "low_resolution_size"
    )
    high_size = _positive_int(
        configuration.get("high_resolution_size"), "high_resolution_size"
    )
    if (low_size, high_size) != (32, 128):
        raise ValueError(
            "Checkpoint must be configured for 32x32 inputs and 128x128 outputs"
        )
    _positive_int(configuration.get("residual_blocks", 16), "residual_blocks")
    return configuration


def _validate_class_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or len(value) != 2:
        raise ValueError("Checkpoint class_mapping must contain exactly two classes")
    mapping = {str(name): int(label) for name, label in value.items()}
    if set(mapping.values()) != {0, 1} or len(mapping) != 2:
        raise ValueError("Checkpoint class_mapping must map two classes to 0 and 1")
    return mapping


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_sources(records: pd.DataFrame, raw_directory: Path) -> None:
    for source in records["filepath"]:
        relative = Path(str(source))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Training manifest filepath must be relative: {source}")
        path = raw_directory / relative
        if not path.is_file():
            raise FileNotFoundError(f"Training source image not found: {path}")


def _generated_path(output: Path, source: str, class_name: str) -> Path:
    class_path = Path(class_name)
    if class_path.name != class_name or class_name in {".", ".."}:
        raise ValueError(f"Invalid class name in training manifest: {class_name}")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    stem = Path(source).stem
    return output / class_name / f"{stem}-{digest}.png"


def _provenance(
    checkpoint: GeneratorCheckpoint, manifest_path: Path
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_identifier": checkpoint.identifier,
        "checkpoint_filename": checkpoint.path.name,
        "checkpoint_epoch": checkpoint.epoch,
        "configuration": checkpoint.configuration,
        "class_mapping": checkpoint.class_mapping,
        "source_manifest": manifest_path.name,
        "low_resolution_size": 32,
        "high_resolution_size": 128,
        "output_range": "[-1, 1] before PNG conversion to [0, 1]",
    }


def _validate_existing_provenance(
    output_directory: Path,
    metadata_path: Path,
    expected: Mapping[str, Any],
) -> None:
    existing_files = (
        [path for path in output_directory.rglob("*") if path.is_file()]
        if output_directory.is_dir()
        else []
    )
    if not metadata_path.is_file():
        if existing_files:
            raise ValueError(
                "Generated output directory has files but no checkpoint provenance; "
                f"use a new directory or remove it deliberately: {output_directory}"
            )
        return
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid generation metadata: {metadata_path}") from error
    if existing != dict(expected):
        raise ValueError(
            "Generated images belong to a different checkpoint or configuration; "
            f"refusing to overwrite {output_directory}"
        )


def _manifest_record(
    source_row: pd.Series, generated_path: Path, manifest_parent: Path
) -> dict[str, str | int]:
    try:
        relative = generated_path.relative_to(manifest_parent).as_posix()
    except ValueError as error:
        raise ValueError(
            "Generated images must be within the generated manifest directory"
        ) from error
    return {
        "filepath": relative,
        "source_filepath": str(source_row["filepath"]),
        "class_name": str(source_row["class_name"]),
        "label": int(source_row["label"]),
    }


def _validate_generated_batch(images: Tensor, expected_batch_size: int) -> None:
    if images.shape != (expected_batch_size, 3, 128, 128):
        raise ValueError(
            "Generator must return RGB tensors shaped (N, 3, 128, 128); "
            f"received {tuple(images.shape)}"
        )
    if not torch.isfinite(images).all():
        raise ValueError("Generator returned non-finite pixel values")
    tolerance = 1e-5
    if images.min().item() < -1.0 - tolerance or images.max().item() > 1.0 + tolerance:
        raise ValueError("Generator output must remain in the documented [-1, 1] range")


def _save_generated_image(tensor: Tensor, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    image = F.to_pil_image(denormalize_srgan(tensor)).convert("RGB")
    image.save(temporary, format="PNG")
    if not _is_valid_generated_image(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Generated image failed validation: {destination}")
    temporary.replace(destination)


def _is_valid_generated_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode == "RGB" and image.size == (128, 128)
    except (OSError, UnidentifiedImageError):
        return False


def _write_manifest(records: Mapping[int, dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [records[index] for index in sorted(records)],
        columns=GENERATED_MANIFEST_COLUMNS,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
