from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from cs2_round_predictor.config import (
    DEFAULT_CORE_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    PROCESSED_DATA_DIR,
)
from cs2_round_predictor.datasets import ensure_default_core_dataset
from cs2_round_predictor.models.predict import predict_round_probabilities


DEFAULT_OUTPUT_CSV = PROCESSED_DATA_DIR / "round_predictions.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict T-side round win probabilities for a core feature dataset."
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=DEFAULT_CORE_DATASET_PATH,
        help="Path to the core round feature dataset CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to save the per-round predictions CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_csv = args.dataset_csv
    if dataset_csv == DEFAULT_CORE_DATASET_PATH:
        dataset_csv = ensure_default_core_dataset()

    dataset = pd.read_csv(dataset_csv)
    predictions = predict_round_probabilities(dataset, model_path=args.model_path)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)

    print(f"Loaded core dataset from {dataset_csv}")
    print(f"Predicted rounds: {len(predictions)}")
    print(f"Saved predictions to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
