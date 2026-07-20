"""Generate Model B training images from a completed SRGAN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from applied_ai_midterm.generation import generate_model_b_dataset
from applied_ai_midterm.training import select_device


def parse_args() -> argparse.Namespace:
    """Parse generation paths and execution settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--train-manifest", type=Path, default=Path("data/splits/train.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated/model_b_train")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/generated/model_b_train.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate the plan or create the resumable generated dataset."""
    args = parse_args()
    device = select_device() if args.device == "auto" else torch.device(args.device)
    print(f"Selected device: {device}")
    summary = generate_model_b_dataset(
        args.checkpoint,
        raw_directory=args.raw_dir,
        train_manifest_path=args.train_manifest,
        output_directory=args.output_dir,
        output_manifest_path=args.output_manifest,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    action = "would generate" if summary.dry_run else "generated"
    print(
        f"{action} {summary.generated} images; skipped {summary.skipped} valid "
        f"images out of {summary.total} records"
    )
    print(f"Checkpoint SHA-256: {summary.checkpoint_identifier}")
    print(f"Manifest: {summary.manifest_path}")


if __name__ == "__main__":
    main()
