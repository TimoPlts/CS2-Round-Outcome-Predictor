from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cs2_round_predictor.config import DEFAULT_NEURAL_MODEL_PATH
from cs2_round_predictor.features.core_features import (
    CORE_FEATURE_COLUMNS,
    MATCH_ID_COLUMN,
    TARGET_COLUMN,
)

GROUP_TEST_SIZE = 0.25
TRAIN_VALIDATION_SIZE = 0.2
MAX_SPLIT_ATTEMPTS = 25


@dataclass(slots=True)
class NeuralTrainingResult:
    accuracy: float
    roc_auc: float
    log_loss_value: float
    train_rows: int
    validation_rows: int
    test_rows: int
    train_matches: int
    test_matches: int
    best_epoch: int
    epochs_completed: int
    decision_threshold: float
    duplicate_matches_removed: int
    duplicate_rows_removed: int
    device: str
    model_path: Path
    resumed_from: Path | None


class RoundOutcomeMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_sizes: Sequence[int] = (16,),
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_size
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def train_neural_model(
    dataset: pd.DataFrame,
    *,
    model_path: Path = DEFAULT_NEURAL_MODEL_PATH,
    resume_from: Path | None = None,
    hidden_sizes: Sequence[int] = (16,),
    dropout: float = 0.4,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 200,
    random_state: int = 42,
    device: str | None = None,
    epoch_reporter: Callable[[dict[str, float | int]], None] | None = None,
) -> NeuralTrainingResult:
    working, duplicate_matches_removed, duplicate_rows_removed = _prepare_dataset(dataset)
    X = working.loc[:, CORE_FEATURE_COLUMNS]
    y = working[TARGET_COLUMN].astype(int)
    groups = working[MATCH_ID_COLUMN].astype(str)

    train_idx, test_idx = _build_group_split(
        X,
        y,
        groups,
        test_size=GROUP_TEST_SIZE,
        random_state=random_state,
    )

    X_train_full = X.iloc[train_idx].reset_index(drop=True)
    y_train_full = y.iloc[train_idx].reset_index(drop=True)
    groups_train_full = groups.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    train_matches = groups.iloc[train_idx].nunique()
    test_matches = groups.iloc[test_idx].nunique()

    train_split_idx, validation_split_idx = _build_group_validation_split(
        X_train_full,
        y_train_full,
        groups_train_full,
        validation_size=TRAIN_VALIDATION_SIZE,
        random_state=random_state,
    )
    X_train = X_train_full.iloc[train_split_idx].reset_index(drop=True)
    y_train = y_train_full.iloc[train_split_idx].reset_index(drop=True)
    X_validation = X_train_full.iloc[validation_split_idx].reset_index(drop=True)
    y_validation = y_train_full.iloc[validation_split_idx].reset_index(drop=True)

    normalization = _fit_normalization(X_train)
    X_train_tensor = _frame_to_tensor(X_train, normalization)
    X_validation_tensor = _frame_to_tensor(X_validation, normalization)
    X_test_tensor = _frame_to_tensor(X_test, normalization)
    y_train_tensor = torch.tensor(y_train.to_numpy(dtype="float32"))
    y_validation_tensor = torch.tensor(y_validation.to_numpy(dtype="float32"))
    y_test_tensor = torch.tensor(y_test.to_numpy(dtype="float32"))

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(random_state)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(random_state)

    resume_checkpoint = load_neural_checkpoint(resume_from) if resume_from else None
    checkpoint_hidden_sizes = _coerce_hidden_sizes(
        resume_checkpoint["hidden_sizes"] if resume_checkpoint else hidden_sizes
    )
    checkpoint_dropout = (
        float(resume_checkpoint["dropout"]) if resume_checkpoint else float(dropout)
    )

    model = RoundOutcomeMLP(
        input_dim=len(CORE_FEATURE_COLUMNS),
        hidden_sizes=checkpoint_hidden_sizes,
        dropout=checkpoint_dropout,
    ).to(target_device)
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model_state_dict"])

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=min(batch_size, max(1, len(X_train_tensor))),
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_validation_loss = float("inf")
    best_epoch = 0

    epochs_completed = 0
    for epoch in range(1, epochs + 1):
        epochs_completed = epoch
        model.train()
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(target_device)
            batch_targets = batch_targets.to(target_device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        train_metrics = _evaluate_split_metrics(
            model,
            X_train_tensor,
            y_train_tensor,
            criterion,
            target_device,
        )
        validation_metrics = _evaluate_split_metrics(
            model,
            X_validation_tensor,
            y_validation_tensor,
            criterion,
            target_device,
        )
        validation_loss = validation_metrics["loss"]
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch_reporter is not None:
            epoch_reporter(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "validation_loss": validation_metrics["loss"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_roc_auc": validation_metrics["roc_auc"],
                    "best_epoch_so_far": best_epoch,
                }
            )

    model.load_state_dict(best_state)

    validation_probabilities = _predict_probabilities_tensor(
        model,
        X_validation_tensor,
        target_device,
    )
    decision_threshold = _select_accuracy_threshold(y_validation, validation_probabilities)
    probabilities = _predict_probabilities_tensor(model, X_test_tensor, target_device)
    predictions = (probabilities >= decision_threshold).astype(int)

    checkpoint = {
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()}, # geleerde gewichten
        "feature_columns": list(CORE_FEATURE_COLUMNS), # verwachte features
        "normalization_mean": normalization["mean"], # gemiddelden
        "normalization_std": normalization["std"], # standaardafwijkingen
        "hidden_sizes": list(checkpoint_hidden_sizes), # architectuur
        "dropout": checkpoint_dropout,
        "decision_threshold": decision_threshold,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)

    return NeuralTrainingResult(
        accuracy=checkpoint["metrics"]["accuracy"],
        roc_auc=checkpoint["metrics"]["roc_auc"],
        log_loss_value=checkpoint["metrics"]["log_loss"],
        train_rows=len(X_train),
        validation_rows=len(X_validation),
        test_rows=len(X_test),
        train_matches=int(train_matches),
        test_matches=int(test_matches),
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        decision_threshold=decision_threshold,
        duplicate_matches_removed=duplicate_matches_removed,
        duplicate_rows_removed=duplicate_rows_removed,
        device=str(target_device),
        model_path=model_path,
        resumed_from=resume_from,
    )


def predict_round_probabilities_neural(
    dataset: pd.DataFrame,
    *,
    model_path: str | Path,
    device: str | None = None,
) -> pd.DataFrame:
    checkpoint = load_neural_checkpoint(model_path)
    missing_columns = sorted(set(CORE_FEATURE_COLUMNS).difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required feature columns: " + ", ".join(missing_columns)
        )

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RoundOutcomeMLP(
        input_dim=len(CORE_FEATURE_COLUMNS),
        hidden_sizes=_coerce_hidden_sizes(checkpoint["hidden_sizes"]),
        dropout=float(checkpoint["dropout"]),
    ).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])

    normalization = {
        "mean": checkpoint["normalization_mean"],
        "std": checkpoint["normalization_std"],
    }
    features_tensor = _frame_to_tensor(dataset.loc[:, CORE_FEATURE_COLUMNS], normalization)
    probabilities = _predict_probabilities_tensor(model, features_tensor, target_device)
    decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
    predictions = (probabilities >= decision_threshold).astype(int)

    output = dataset.copy()
    output["t_win_probability"] = probabilities
    output["predicted_won_round"] = predictions
    output["predicted_winner"] = output["predicted_won_round"].map({1: "T", 0: "CT"})
    return output


def load_neural_checkpoint(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Neural model checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    required_keys = {
        "model_state_dict",
        "feature_columns",
        "normalization_mean",
        "normalization_std",
        "hidden_sizes",
        "dropout",
    }
    missing_keys = sorted(required_keys.difference(checkpoint))
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required keys: " + ", ".join(missing_keys)
        )
    feature_columns = list(checkpoint["feature_columns"])
    if feature_columns != list(CORE_FEATURE_COLUMNS):
        raise ValueError(
            "Checkpoint feature columns do not match the current core feature schema."
        )
    return checkpoint


def _prepare_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    missing_columns = sorted(
        set(CORE_FEATURE_COLUMNS + [MATCH_ID_COLUMN, TARGET_COLUMN]).difference(dataset.columns)
    )
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    working = dataset.loc[:, [MATCH_ID_COLUMN, *CORE_FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    working, duplicate_matches_removed, duplicate_rows_removed = _drop_duplicate_matches(working)
    if len(working) < 10:
        raise ValueError("Dataset is too small to train reliably. Parse more demos first.")

    unique_matches = working[MATCH_ID_COLUMN].astype(str).nunique()
    if unique_matches < 2:
        raise ValueError(
            "Need at least 2 unique `match_id` values for match-based evaluation. Parse more demos first."
        )

    if working[TARGET_COLUMN].astype(int).nunique() < 2:
        raise ValueError("Need both classes in `won_round` to train a classifier.")

    return working, duplicate_matches_removed, duplicate_rows_removed


def _drop_duplicate_matches(dataset: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    content_columns = [*CORE_FEATURE_COLUMNS, TARGET_COLUMN]
    seen_signatures: set[str] = set()
    duplicate_match_ids: list[str] = []

    for match_id, match_rows in dataset.groupby(MATCH_ID_COLUMN, sort=False):
        match_content = match_rows.loc[:, content_columns].reset_index(drop=True)
        signature_key = hashlib.sha256(
            match_content.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        if signature_key in seen_signatures:
            duplicate_match_ids.append(str(match_id))
        else:
            seen_signatures.add(signature_key)

    duplicate_mask = dataset[MATCH_ID_COLUMN].astype(str).isin(duplicate_match_ids)
    return (
        dataset.loc[~duplicate_mask].reset_index(drop=True),
        len(duplicate_match_ids),
        int(duplicate_mask.sum()),
    )


def _build_group_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    splitter = GroupShuffleSplit(
        n_splits=MAX_SPLIT_ATTEMPTS,
        test_size=test_size,
        random_state=random_state,
    )

    for train_idx, test_idx in splitter.split(X, y, groups):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        return list(train_idx), list(test_idx)

    raise ValueError(
        "Could not create a match-based train/test split with both classes in each split. "
        "Parse more demos with more varied round outcomes."
    )


def _build_group_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    validation_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    splitter = GroupShuffleSplit(
        n_splits=MAX_SPLIT_ATTEMPTS,
        test_size=validation_size,
        random_state=random_state,
    )
    for train_idx, validation_idx in splitter.split(X, y, groups):
        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]
        if y_train.nunique() < 2 or y_validation.nunique() < 2:
            continue
        return list(train_idx), list(validation_idx)

    raise ValueError(
        "Could not create a match-based train/validation split with both classes in each split. "
        "Parse more demos with more varied round outcomes."
    )


def _select_accuracy_threshold(targets: pd.Series, probabilities: Any) -> float:
    candidates = [round(value / 100, 2) for value in range(30, 71)]
    scored = [
        (float(accuracy_score(targets, probabilities >= threshold)), threshold)
        for threshold in candidates
    ]
    best_accuracy = max(score for score, _ in scored)
    best_thresholds = [
        threshold for score, threshold in scored if score == best_accuracy
    ]
    return min(best_thresholds, key=lambda threshold: abs(threshold - 0.5))


def _fit_normalization(frame: pd.DataFrame) -> dict[str, list[float]]:
    mean = frame.mean(axis=0).astype("float32")
    std = frame.std(axis=0, ddof=0).replace(0, 1).astype("float32")
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _frame_to_tensor(frame: pd.DataFrame, normalization: dict[str, list[float]]) -> torch.Tensor:
    values = frame.to_numpy(dtype="float32")
    mean = torch.tensor(normalization["mean"], dtype=torch.float32)
    std = torch.tensor(normalization["std"], dtype=torch.float32)
    return (torch.tensor(values) - mean) / std


def _evaluate_split_metrics(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        loss = criterion(logits, targets.to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (probabilities >= 0.5).astype(int)
    target_values = targets.cpu().numpy().astype(int)
    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy_score(target_values, predictions)),
        "roc_auc": _safe_roc_auc(target_values, probabilities),
    }


def _predict_probabilities_tensor(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
) -> Any:
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return probabilities


def _coerce_hidden_sizes(raw_hidden_sizes: Sequence[int] | Any) -> tuple[int, ...]:
    hidden_sizes = tuple(int(size) for size in raw_hidden_sizes)
    if not hidden_sizes:
        raise ValueError("At least one hidden layer size is required.")
    return hidden_sizes


def _safe_roc_auc(targets: Any, probabilities: Any) -> float:
    unique_values = set(int(value) for value in targets)
    if len(unique_values) < 2:
        return float("nan")
    return float(roc_auc_score(targets, probabilities))
