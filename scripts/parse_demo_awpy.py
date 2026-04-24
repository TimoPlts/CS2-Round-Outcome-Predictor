from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import typer

from cs2_round_predictor.config import DEFAULT_DATASET_PATH, RAW_DATA_DIR
from cs2_round_predictor.parsing.demo_parser import (
    build_round_dataset_from_artifacts,
    export_artifacts_to_csv,
    parse_demo_file,
)

app = typer.Typer()


@app.command()
def main(
    demo_path: Path,
    raw_output_dir: Path | None = None,
    processed_output_csv: Path | None = None,
    verbose: bool = False,
) -> None:
    raw_dir = raw_output_dir or (RAW_DATA_DIR / demo_path.stem)
    processed_csv = processed_output_csv or DEFAULT_DATASET_PATH

    artifacts = parse_demo_file(demo_path, verbose=verbose)
    export_artifacts_to_csv(artifacts, raw_dir)

    round_dataset = build_round_dataset_from_artifacts(
        artifacts,
        match_id=demo_path.stem,
    )
    processed_csv.parent.mkdir(parents=True, exist_ok=True)
    round_dataset.to_csv(processed_csv, index=False)

    typer.echo(f"Saved raw Awpy tables to {raw_dir}")
    typer.echo(f"Saved round-level dataset to {processed_csv}")


if __name__ == "__main__":
    app()
