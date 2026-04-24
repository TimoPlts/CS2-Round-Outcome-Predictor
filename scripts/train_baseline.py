from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import typer

from cs2_round_predictor.config import DEFAULT_DATASET_PATH
from cs2_round_predictor.models.train import train_baseline_model

app = typer.Typer()


@app.command()
def main(dataset_csv: Path = DEFAULT_DATASET_PATH) -> None:
    dataset = pd.read_csv(dataset_csv)
    result = train_baseline_model(dataset)
    typer.echo(f"Accuracy: {result.accuracy:.3f}")
    typer.echo(f"ROC-AUC: {result.roc_auc:.3f}")
    typer.echo(f"Log loss: {result.log_loss_value:.3f}")


if __name__ == "__main__":
    app()
