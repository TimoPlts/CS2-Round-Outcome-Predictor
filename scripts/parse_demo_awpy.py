from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cs2_round_predictor.config import (
    DEFAULT_CORE_DATASET_PATH,
    DEFAULT_DATASET_PATH,
    RAW_DATA_DIR,
)
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    demo_path = args.demo_path
    if not demo_path.exists():
        parser.error(f"Demo file does not exist: {demo_path}")

    raw_dir = args.raw_output_dir or (RAW_DATA_DIR / demo_path.stem)
    processed_csv = args.processed_output_csv or DEFAULT_DATASET_PATH
    core_csv = args.core_output_csv or DEFAULT_CORE_DATASET_PATH

    artifacts = parse_demo_file(demo_path, verbose=args.verbose)
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

    print(f"Saved raw Awpy tables to {raw_dir}")
    print(f"Saved round-level dataset to {processed_csv}")
    print(f"Saved core round dataset to {core_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
