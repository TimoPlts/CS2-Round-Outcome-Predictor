from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from cs2_round_predictor.config import DEFAULT_CORE_DATASET_PATH, DEFAULT_NEURAL_MODEL_PATH
from cs2_round_predictor.datasets import ensure_default_core_dataset
from cs2_round_predictor.models.neural import train_neural_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch round outcome model on the core round feature dataset."
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
        default=DEFAULT_NEURAL_MODEL_PATH,
        help="Where to save the trained neural model checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from the checkpoint stored at --model-path.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience measured in epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate for hidden layers.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override, for example cpu or cuda.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dataset_csv = args.dataset_csv
    if dataset_csv == DEFAULT_CORE_DATASET_PATH:
        dataset_csv = ensure_default_core_dataset()

    resume_from = args.model_path if args.resume else None
    if args.resume and not args.model_path.exists():
        parser.error(f"Cannot resume because checkpoint does not exist: {args.model_path}")

    dataset = pd.read_csv(dataset_csv)
    result = train_neural_model(
        dataset,
        model_path=args.model_path,
        resume_from=resume_from,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
    )

    print(f"Loaded core dataset from {dataset_csv}")
    print(f"Train rows: {result.train_rows}")
    print(f"Validation rows: {result.validation_rows}")
    print(f"Test rows: {result.test_rows}")
    print(f"Train matches: {result.train_matches}")
    print(f"Test matches: {result.test_matches}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Epochs completed: {result.epochs_completed}")
    print(f"Accuracy: {result.accuracy:.3f}")
    print(f"ROC-AUC: {result.roc_auc:.3f}")
    print(f"Log loss: {result.log_loss_value:.3f}")
    print(f"Device: {result.device}")
    if result.resumed_from is not None:
        print(f"Resumed from: {result.resumed_from}")
    print(f"Saved neural checkpoint to {result.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
