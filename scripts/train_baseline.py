from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from cs2_round_predictor.config import DEFAULT_CORE_DATASET_PATH, DEFAULT_MODEL_PATH
from cs2_round_predictor.datasets import ensure_default_core_dataset
from cs2_round_predictor.models.train import train_baseline_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a simple baseline round outcome model."
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
        help="Where to save the trained model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_csv = args.dataset_csv
    if dataset_csv == DEFAULT_CORE_DATASET_PATH:
        dataset_csv = ensure_default_core_dataset()

    dataset = pd.read_csv(dataset_csv)
    result = train_baseline_model(dataset, model_path=args.model_path)

    print(f"Loaded core dataset from {dataset_csv}")
    print(f"Train rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    print(f"Accuracy: {result.accuracy:.3f}")
    print(f"ROC-AUC: {result.roc_auc:.3f}")
    print(f"Log loss: {result.log_loss_value:.3f}")
    print(f"Saved model to {result.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
