from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_round_predictor.config import (
    demo_core_features_path,
    demo_raw_artifacts_dir,
    demo_round_features_path,
    demo_source_path,
)
from cs2_round_predictor.datasets import sync_default_datasets
from cs2_round_predictor.features.core_features import build_core_feature_table
from cs2_round_predictor.parsing.demo_parser import (
    build_round_dataset_from_artifacts,
    export_artifacts_to_csv,
    parse_demo_file,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse one CS2 demo with Awpy and export raw tables plus full and core round datasets."
    )
    parser.add_argument("demo_path", type=Path, help="Path to a .dem file.")
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=None,
        help="Directory for raw parsed Awpy tables.",
    )
    parser.add_argument(
        "--processed-output-csv",
        type=Path,
        default=None,
        help="Path for the full processed round dataset CSV.",
    )
    parser.add_argument(
        "--core-output-csv",
        type=Path,
        default=None,
        help="Path for the compact core round dataset CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable Awpy verbose parsing logs.",
    )
    return parser


def resolve_demo_path(demo_path: Path) -> Path:
    if not demo_path.exists() and not demo_path.is_absolute():
        candidate = demo_source_path(demo_path.name)
        if candidate.exists():
            return candidate
    return demo_path


def parse_and_export_demo(
    demo_path: Path,
    *,
    raw_output_dir: Path | None = None,
    processed_output_csv: Path | None = None,
    core_output_csv: Path | None = None,
    verbose: bool = False,
    sync_defaults: bool = True,
) -> dict[str, Path | None]:
    demo_path = resolve_demo_path(demo_path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file does not exist: {demo_path}")

    raw_dir = raw_output_dir or demo_raw_artifacts_dir(demo_path.stem)
    processed_csv = processed_output_csv or demo_round_features_path(demo_path.stem)
    core_csv = core_output_csv or demo_core_features_path(demo_path.stem)

    artifacts = parse_demo_file(demo_path, verbose=verbose)
    export_artifacts_to_csv(artifacts, raw_dir)

    round_dataset = build_round_dataset_from_artifacts(
        artifacts,
        match_id=demo_path.stem,
    )
    core_dataset = build_core_feature_table(round_dataset)
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    core_csv.parent.mkdir(parents=True, exist_ok=True)
    round_dataset.to_csv(processed_csv, index=False)
    core_dataset.to_csv(core_csv, index=False)

    aggregate_round_csv: Path | None = None
    aggregate_core_csv: Path | None = None
    if sync_defaults:
        aggregate_round_csv, aggregate_core_csv = sync_default_datasets()

    return {
        "demo_path": demo_path,
        "raw_dir": raw_dir,
        "processed_csv": processed_csv,
        "core_csv": core_csv,
        "aggregate_round_csv": aggregate_round_csv,
        "aggregate_core_csv": aggregate_core_csv,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    demo_path = resolve_demo_path(args.demo_path)

    if not demo_path.exists():
        parser.error(f"Demo file does not exist: {demo_path}")

    result = parse_and_export_demo(
        demo_path,
        raw_output_dir=args.raw_output_dir,
        processed_output_csv=args.processed_output_csv,
        core_output_csv=args.core_output_csv,
        verbose=args.verbose,
        sync_defaults=True,
    )

    print(f"Source demo: {result['demo_path']}")
    print(f"Saved raw Awpy tables to {result['raw_dir']}")
    print(f"Saved round-level dataset to {result['processed_csv']}")
    print(f"Saved core round dataset to {result['core_csv']}")
    if result["aggregate_round_csv"] is not None:
        print(f"Updated default round dataset to {result['aggregate_round_csv']}")
    if result["aggregate_core_csv"] is not None:
        print(f"Updated default core dataset to {result['aggregate_core_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
