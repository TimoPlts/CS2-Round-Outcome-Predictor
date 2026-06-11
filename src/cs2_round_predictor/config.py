from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RAW_DEMOS_DIR = RAW_DATA_DIR / "demos"
RAW_PARSED_DIR = RAW_DATA_DIR / "parsed"
PROCESSED_DEMOS_DIR = PROCESSED_DATA_DIR / "by_demo"
DEFAULT_DATASET_PATH = PROCESSED_DATA_DIR / "round_features.csv"
DEFAULT_CORE_DATASET_PATH = PROCESSED_DATA_DIR / "core_round_features.csv"
DEFAULT_NEURAL_MODEL_PATH = PROJECT_ROOT / "models" / "round_outcome_mlp.pt"
DEFAULT_EXPERIMENT_LOG_PATH = EXPERIMENTS_DIR / "experiment_log.jsonl"
ROUND_FEATURES_FILENAME = "round_features.csv"
CORE_FEATURES_FILENAME = "core_round_features.csv"


def demo_source_path(demo_name: str) -> Path: # volledige bestandsnaam
    return RAW_DEMOS_DIR / demo_name


def demo_raw_artifacts_dir(demo_stem: str) -> Path: # bestandsnaam zonder extensie
    return RAW_PARSED_DIR / demo_stem


def demo_processed_dir(demo_stem: str) -> Path:
    return PROCESSED_DEMOS_DIR / demo_stem


def demo_round_features_path(demo_stem: str) -> Path: # data/processed/by_demo/match1
    return demo_processed_dir(demo_stem) / ROUND_FEATURES_FILENAME # data/processed/by_demo/match1/round_features.csv


def demo_core_features_path(demo_stem: str) -> Path:
    return demo_processed_dir(demo_stem) / CORE_FEATURES_FILENAME


def demo_round_feature_paths() -> list[Path]: # om 1 dataset te maken, daarna panda inlezen en combineren
    if not PROCESSED_DEMOS_DIR.exists():
        return []
    return sorted(
        path for path in PROCESSED_DEMOS_DIR.glob(f"*/{ROUND_FEATURES_FILENAME}") if path.is_file() # data/processed/by_demo/*/round_features.csv
    )


def demo_core_feature_paths() -> list[Path]: # ==
    if not PROCESSED_DEMOS_DIR.exists():
        return []
    return sorted(
        path for path in PROCESSED_DEMOS_DIR.glob(f"*/{CORE_FEATURES_FILENAME}") if path.is_file()
    )
