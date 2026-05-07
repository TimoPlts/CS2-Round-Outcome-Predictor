# CS2 Round Outcome Predictor

This project predicts whether the T side will win a CS2 round using pre-round features extracted from `.dem` files.

The current pipeline already works end-to-end:
- parse CS2 demos with Awpy
- build round-level datasets
- train a baseline logistic regression model
- train a PyTorch MLP model
- generate per-round win probabilities

At the moment, the repository contains 28 demo files and an aggregated dataset of 610 rounds.

## What Is In The Repo

- `scripts/parse_demo.py`: parse one demo and export raw + processed CSV files
- `scripts/parse_all_demos.py`: batch-parse all demos in `data/raw/demos`
- `scripts/train_baseline.py`: train the scikit-learn baseline model
- `scripts/train_neural.py`: train the PyTorch neural network
- `scripts/predict_rounds.py`: create predictions with the baseline model
- `scripts/predict_neural.py`: create predictions with the neural model
- `data/processed/`: aggregated datasets and prediction outputs
- `models/`: saved baseline and neural model files

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

Train the baseline model:

```bash
python scripts/train_baseline.py
```

Train the neural model:

```bash
python scripts/train_neural.py
```

Generate predictions:

```bash
python scripts/predict_rounds.py
python scripts/predict_neural.py
```

## Output Files

- `data/processed/round_features.csv`: full aggregated round dataset
- `data/processed/core_round_features.csv`: compact feature set used for training
- `data/processed/round_predictions.csv`: baseline model predictions
- `data/processed/round_predictions_neural.csv`: neural model predictions
- `models/baseline_model.joblib`: saved logistic regression model
- `models/round_outcome_mlp.pt`: saved neural network checkpoint

## Notes

- Evaluation is match-aware: training and test splits are grouped by `match_id`
- The project currently focuses on offline demo analysis, not live in-game prediction
- The repository is script-first right now; there is no finished dashboard in this version
