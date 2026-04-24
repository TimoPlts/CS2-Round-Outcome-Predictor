from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import typer

from cs2_round_predictor.config import DEFAULT_DATASET_PATH
from cs2_round_predictor.features.round_features import build_round_feature_table
from cs2_round_predictor.parsing.demo_parser import load_rounds_csv

app = typer.Typer()


@app.command()
def main(input_csv: Path, output_csv: Path = DEFAULT_DATASET_PATH) -> None:
    parsed = load_rounds_csv(input_csv)
    dataset = build_round_feature_table(parsed.rounds)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_csv, index=False)
    typer.echo(f"Saved dataset to {output_csv}")


if __name__ == "__main__":
    app()
