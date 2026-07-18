"""Create or validate the project's persistent train/test manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from applied_ai_midterm.config import DEFAULT_CONFIG_PATH, load_config
from applied_ai_midterm.data import prepare_splits


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and intentional split-replacement options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deliberately replace existing split manifests.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare the split once and print an audit-friendly count summary."""
    args = parse_args()
    config = load_config(args.config)
    train_manifest, test_manifest = prepare_splits(
        args.raw_dir,
        args.split_dir,
        train_ratio=config.train_ratio,
        random_seed=config.random_seed,
        force=args.force,
    )

    print(f"Train manifest: {args.split_dir / 'train.csv'}")
    print(train_manifest.groupby(["class_name", "label"]).size().to_string())
    print(f"Total training images: {len(train_manifest)}")
    print(f"Test manifest: {args.split_dir / 'test.csv'}")
    print(test_manifest.groupby(["class_name", "label"]).size().to_string())
    print(f"Total test images: {len(test_manifest)}")


if __name__ == "__main__":
    main()
