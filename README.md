# CS2 Round Outcome Predictor

This project predicts whether the T side will win a CS2 round using pre-round features extracted from `.dem` files.

The current pipeline already works end-to-end:
- parse CS2 demos with Awpy
- build round-level datasets
- train a PyTorch MLP model
- generate per-round win probabilities

At the moment, the repository contains 28 demo files and an aggregated dataset of 610 rounds.

## What Is In The Repo

- `scripts/parse_demo.py`: parse one demo and export raw + processed CSV files
- `scripts/parse_all_demos.py`: batch-parse all demos in `data/raw/demos`
- `scripts/train_neural.py`: train the PyTorch neural network
- `scripts/predict_neural.py`: create predictions with the neural model
- `scripts/download_faceit_demos.py`: search FACEIT match history and build a demo manifest for specific players and maps
- `streamlit_app.py`: round-by-round web UI with team inputs, real result, and neural prediction
- `data/processed/`: aggregated datasets and prediction outputs
- `models/`: saved neural model files

## Core Features

The compact training dataset uses round-level features such as:
- pistol round flag
- previous round winner
- win streak difference
- CT defuse kits
- money difference
- equipment value difference
- armor difference
- helmet difference
- utility difference
- smoke difference
- flash difference

Target:
- `1` = T side won the round
- `0` = CT side won the round

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Parse all demos:

```bash
python scripts/parse_all_demos.py
```

Train the neural model:

```bash
python scripts/train_neural.py
```

Generate predictions:

```bash
python scripts/predict_neural.py
```

Start the web UI:

```bash
streamlit run streamlit_app.py
```

FACEIT demo search for one player and one map:

```bash
export FACEIT_DATA_API_KEY="your_faceit_data_api_key"
python scripts/download_faceit_demos.py --players verso-topo --map mirage --list-only
```

## Output Files

- `data/processed/round_features.csv`: full aggregated round dataset
- `data/processed/core_round_features.csv`: compact feature set used for training
- `data/processed/round_predictions_neural.csv`: neural model predictions
- `models/round_outcome_mlp.pt`: saved neural network checkpoint

## Notes

- Evaluation is match-aware: training and test splits are grouped by `match_id`
- The project currently focuses on offline demo analysis, not live in-game prediction
- The repository is script-first right now; there is no finished dashboard in this version
- FACEIT match lookup already works with a normal FACEIT Data API key and can build a manifest of matching demo resources
- Downloading the actual FACEIT demo files still requires a separate `FACEIT_DOWNLOADS_API_TOKEN`
- That download token is currently not available in this project setup, so FACEIT demo collection is paused at the manifest stage
