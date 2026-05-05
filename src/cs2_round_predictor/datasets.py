from __future__ import annotations

from pathlib import Path

import pandas as pd

from cs2_round_predictor.config import (
    DEFAULT_CORE_DATASET_PATH,
    DEFAULT_DATASET_PATH,
    demo_core_feature_paths,
    demo_round_feature_paths,
)
from cs2_round_predictor.features.core_features import build_core_feature_table


def ensure_default_round_dataset() -> Path:
    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH

    dataset_path = refresh_default_round_dataset()
    if dataset_path is None:
        raise FileNotFoundError(
            f"Round dataset not found at {DEFAULT_DATASET_PATH} and no per-demo files exist under by_demo/."
        )
    return dataset_path


def ensure_default_core_dataset() -> Path:
    if DEFAULT_CORE_DATASET_PATH.exists():
        return DEFAULT_CORE_DATASET_PATH

    dataset_path = refresh_default_core_dataset()
    if dataset_path is None:
        raise FileNotFoundError(
            f"Core dataset not found at {DEFAULT_CORE_DATASET_PATH} and no per-demo files exist under by_demo/."
        )
    return dataset_path


def refresh_default_round_dataset() -> Path | None:
    dataset = _combine_csv_files(demo_round_feature_paths())
    if dataset is None:
        return None

    DEFAULT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DEFAULT_DATASET_PATH, index=False)
    return DEFAULT_DATASET_PATH


def refresh_default_core_dataset() -> Path | None:
    core_dataset = _combine_csv_files(demo_core_feature_paths())
    if core_dataset is None:
        round_dataset_path = refresh_default_round_dataset()
        if round_dataset_path is None:
            return None
        core_dataset = build_core_feature_table(pd.read_csv(round_dataset_path))

    DEFAULT_CORE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    core_dataset.to_csv(DEFAULT_CORE_DATASET_PATH, index=False)
    return DEFAULT_CORE_DATASET_PATH


def sync_default_datasets() -> tuple[Path | None, Path | None]:
    round_dataset_path = refresh_default_round_dataset()
    core_dataset_path = refresh_default_core_dataset()
    return round_dataset_path, core_dataset_path


def _combine_csv_files(paths: list[Path]) -> pd.DataFrame | None:
    if not paths:
        return None

    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)
