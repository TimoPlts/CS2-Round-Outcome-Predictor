from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_DATASET_PATH = PROCESSED_DATA_DIR / "round_features.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"
