from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from cs2_round_predictor.config import DEFAULT_CORE_DATASET_PATH, DEFAULT_DATASET_PATH
from cs2_round_predictor.features.core_features import (
    TARGET_COLUMN,
    build_core_feature_table,
    rank_core_features,
)


DEFAULT_INPUT_CSV = DEFAULT_DATASET_PATH
DEFAULT_OUTPUT_CSV = DEFAULT_CORE_DATASET_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a compact core-feature dataset and rank the strongest signals."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the full round feature dataset CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path for the compact core feature dataset CSV.",
    )
    return parser
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset = pd.read_csv(args.input_csv)
    core_dataset = build_core_feature_table(dataset)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    core_dataset.to_csv(args.output_csv, index=False)

    ranking = rank_core_features(core_dataset)

    print(f"Saved core feature dataset to {args.output_csv}")
    print(f"Rows: {len(core_dataset)}")
    print(f"T-side wins: {int(core_dataset[TARGET_COLUMN].sum())}")
    print("Recommended feature priority:")
    for feature_name, score in ranking[:10]:
        print(f"- {feature_name}: {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
