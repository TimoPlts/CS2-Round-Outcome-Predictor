from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from cs2_round_predictor.config import (
    DEFAULT_CORE_DATASET_PATH,
    DEFAULT_EXPERIMENT_LOG_PATH,
    DEFAULT_NEURAL_MODEL_PATH,
    EXPERIMENTS_DIR,
)
from cs2_round_predictor.datasets import ensure_default_core_dataset
from cs2_round_predictor.features.core_features import CORE_FEATURE_COLUMNS, MATCH_ID_COLUMN, TARGET_COLUMN
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
        help="Number of training epochs to run.",
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
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional readable name for this experiment run.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Short note about what changed in this experiment.",
    )
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=DEFAULT_EXPERIMENT_LOG_PATH,
        help="JSONL file where experiment metadata and metrics are appended.",
    )
    parser.add_argument(
        "--no-archive-model",
        action="store_true",
        help="Do not copy the trained checkpoint into experiments/models/.",
    )
    return parser


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "run"


def _dataset_fingerprint(dataset: pd.DataFrame) -> str:
    relevant_columns = [MATCH_ID_COLUMN, *CORE_FEATURE_COLUMNS, TARGET_COLUMN]
    stable = dataset.loc[:, relevant_columns].sort_values(relevant_columns).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(stable.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()[:16]


def _write_experiment_record(
    *,
    dataset: pd.DataFrame,
    dataset_csv: Path,
    result,
    args: argparse.Namespace,
) -> Path:
    created_at = datetime.now(timezone.utc)
    run_label = args.run_name or "neural-mlp"
    run_id = f"{created_at.strftime('%Y%m%d-%H%M%S')}-{_slugify(run_label)}"

    archived_model_path = None
    if not args.no_archive_model:
        archive_dir = EXPERIMENTS_DIR / "models"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_model_path = archive_dir / f"{run_id}.pt"
        shutil.copy2(result.model_path, archived_model_path)

    record = {
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "run_name": run_label,
        "notes": args.notes,
        "model_type": "PyTorch MLP",
        "dataset_csv": str(dataset_csv),
        "dataset_rows": int(len(dataset)),
        "dataset_matches": int(dataset[MATCH_ID_COLUMN].astype(str).nunique()),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "target_t_win_rate": float(dataset[TARGET_COLUMN].astype(int).mean()),
        "feature_columns": list(CORE_FEATURE_COLUMNS),
        "feature_count": len(CORE_FEATURE_COLUMNS),
        "hidden_sizes": list(args.hidden_sizes),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "requested_epochs": int(args.epochs),
        "best_epoch": int(result.best_epoch),
        "epochs_completed": int(result.epochs_completed),
        "train_rows": int(result.train_rows),
        "validation_rows": int(result.validation_rows),
        "test_rows": int(result.test_rows),
        "train_matches": int(result.train_matches),
        "test_matches": int(result.test_matches),
        "accuracy": float(result.accuracy),
        "roc_auc": float(result.roc_auc),
        "log_loss": float(result.log_loss_value),
        "device": result.device,
        "model_path": str(result.model_path),
        "archived_model_path": str(archived_model_path) if archived_model_path else "",
        "resumed_from": str(result.resumed_from) if result.resumed_from else "",
    }

    args.experiment_log.parent.mkdir(parents=True, exist_ok=True)
    with args.experiment_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return args.experiment_log


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

    def _print_epoch_report(metrics: dict[str, float | int]) -> None:
        train_auc = metrics["train_roc_auc"]
        validation_auc = metrics["validation_roc_auc"]
        train_auc_text = f"{train_auc:.3f}" if train_auc == train_auc else "nan"
        validation_auc_text = (
            f"{validation_auc:.3f}" if validation_auc == validation_auc else "nan"
        )
        print(
            "Epoch "
            f"{metrics['epoch']:>3}: "
            f"train_loss={metrics['train_loss']:.4f} "
            f"train_acc={metrics['train_accuracy']:.3f} "
            f"train_auc={train_auc_text} | "
            f"val_loss={metrics['validation_loss']:.4f} "
            f"val_acc={metrics['validation_accuracy']:.3f} "
            f"val_auc={validation_auc_text} | "
            f"best_epoch={metrics['best_epoch_so_far']}"
        )

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
        device=args.device,
        epoch_reporter=_print_epoch_report,
    )
    experiment_log_path = _write_experiment_record(
        dataset=dataset,
        dataset_csv=dataset_csv,
        result=result,
        args=args,
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
    print(f"Logged experiment to {experiment_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
