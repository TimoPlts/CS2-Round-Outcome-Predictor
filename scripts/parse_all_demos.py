from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_round_predictor.config import (
    RAW_DEMOS_DIR,
    demo_core_features_path,
    demo_raw_artifacts_dir,
    demo_round_features_path,
)
from cs2_round_predictor.datasets import sync_default_datasets
from parse_demo import parse_and_export_demo


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse all CS2 demos in data/raw/demos and export per-demo plus aggregate datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DEMOS_DIR,
        help="Directory containing .dem files to parse.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse demos even if per-demo outputs already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of demos to parse in this run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable Awpy verbose parsing logs.",
    )
    return parser


def _discover_demo_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input demo directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".dem")


def _is_demo_already_processed(demo_path: Path) -> bool:
    raw_dir = demo_raw_artifacts_dir(demo_path.stem)
    processed_csv = demo_round_features_path(demo_path.stem)
    core_csv = demo_core_features_path(demo_path.stem)
    return raw_dir.exists() and processed_csv.exists() and core_csv.exists()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    demo_paths = _discover_demo_paths(args.input_dir)
    if not demo_paths:
        parser.error(f"No .dem files found in {args.input_dir}")

    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be a positive integer.")
        demo_paths = demo_paths[: args.limit]

    parsed_count = 0
    skipped_count = 0
    failed: list[tuple[Path, str]] = []

    for index, demo_path in enumerate(demo_paths, start=1):
        if not args.force and _is_demo_already_processed(demo_path):
            skipped_count += 1
            print(f"[{index}/{len(demo_paths)}] Skipping already processed demo: {demo_path.name}")
            continue

        print(f"[{index}/{len(demo_paths)}] Parsing {demo_path.name}")
        try:
            result = parse_and_export_demo(
                demo_path,
                verbose=args.verbose,
                sync_defaults=False,
            )
        except Exception as exc:
            failed.append((demo_path, str(exc)))
            print(f"Failed to parse {demo_path.name}: {exc}")
            continue

        parsed_count += 1
        print(f"Saved processed demo dataset to {result['processed_csv']}")

    aggregate_round_csv = None
    aggregate_core_csv = None
    if parsed_count > 0:
        aggregate_round_csv, aggregate_core_csv = sync_default_datasets()

    print("")
    print("Batch parse summary")
    print(f"Discovered demos: {len(demo_paths)}")
    print(f"Parsed demos: {parsed_count}")
    print(f"Skipped demos: {skipped_count}")
    print(f"Failed demos: {len(failed)}")
    if aggregate_round_csv is not None:
        print(f"Updated default round dataset to {aggregate_round_csv}")
    if aggregate_core_csv is not None:
        print(f"Updated default core dataset to {aggregate_core_csv}")

    if failed:
        print("Failed files:")
        for demo_path, error_message in failed:
            print(f"- {demo_path.name}: {error_message}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
