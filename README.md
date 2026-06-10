# CS2 Round Outcome Predictor

An end-to-end deep-learning application that predicts whether the Terrorist side
will win a Counter-Strike 2 round from information available at the start of that
round.

The project parses CS2 `.dem` files with Awpy, extracts round-start economy and
equipment features, trains a PyTorch multilayer perceptron, and presents
predictions and evaluation results in a Streamlit dashboard.

## Project Goal

The model answers one question:

> Based on the game state immediately after freeze time, what is the probability
> that the T side wins the round?

This is a binary classification problem:

- `1`: T wins the round
- `0`: CT wins the round

The project predicts individual rounds, not complete match winners.

## Current Results

The current local dataset and saved model use only Mirage matches.

| Item | Value |
|---|---:|
| Dataset rows | 7,116 rounds |
| Unique match IDs before deduplication | 331 |
| Duplicate matches removed for training | 6 matches / 133 rounds |
| Rows used after deduplication | 6,983 |
| Input features | 21 |
| Model architecture | `21 -> 16 -> 1` |
| Trainable parameters | 369 |
| Decision threshold | 0.55 |
| Held-out test accuracy | 0.659 |
| Held-out test ROC-AUC | 0.704 |
| Held-out test log loss | 0.613 |

The held-out test set contains complete matches that were not used for training,
validation, best-epoch selection, or threshold selection.

## Application

The Streamlit application contains three tabs:

- **Round Explorer**: inspect the exact model inputs, predicted T-win
  probability, predicted winner, and actual result for every round.
- **Evaluation**: inspect saved test metrics, a confusion matrix, probability
  distributions, and results per match.
- **Experiments**: compare training runs, hyperparameters, feature sets, dataset
  fingerprints, and test scores.

## End-to-End Pipeline

```text
CS2 .dem files
    |
    v
Awpy parsing
    |
    v
Round-start snapshots after freeze time
    |
    v
Full and core round-level CSV datasets
    |
    v
Match-aware train / validation / test split
    |
    v
PyTorch MLP training
    |
    v
T-win probabilities and predicted winners
    |
    v
Streamlit dashboard
```

## Requirements

- Python 3.11 or newer
- CS2 `.dem` files to rebuild the dataset
- Enough free disk space for demos and parsed tick data
- CUDA-compatible GPU is optional; training also works on CPU

Main dependencies:

- Awpy
- PyTorch
- Pandas
- scikit-learn
- Streamlit
- Altair

## Quick Start

### 1. Create and activate a virtual environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add demo files

Place one or more CS2 demo files in:

```text
data/raw/demos/
```

Create the directory first if it does not exist:

```bash
mkdir -p data/raw/demos
```

Only files ending in `.dem` are discovered by the batch parser.

### 3. Parse all demos

```bash
python scripts/parse_all_demos.py
```

The parser creates raw Awpy tables, per-demo datasets, and combined datasets.
Already processed demos are skipped automatically.

Useful parser options:

```bash
# Parse at most five demos
python scripts/parse_all_demos.py --limit 5

# Re-parse every discovered demo
python scripts/parse_all_demos.py --force

# Show verbose Awpy parsing output
python scripts/parse_all_demos.py --verbose
```

If parsing is interrupted, run the normal command again. Completed demos are
skipped. Check the last interrupted demo before continuing because the current
skip check verifies that output paths exist, not that every generated file is
complete.

### 4. Train the neural network

```bash
python scripts/train_neural.py \
  --run-name "mirage-mlp" \
  --notes "21 round-start features, one hidden layer"
```

This command:

- loads `data/processed/core_round_features.csv`;
- removes exact duplicate matches;
- creates match-aware train, validation, and test splits;
- normalizes features using training data only;
- trains the PyTorch MLP;
- selects the best epoch using validation loss;
- selects a classification threshold using validation accuracy;
- evaluates once on held-out test matches;
- saves the model checkpoint;
- appends the experiment metadata and scores to the experiment log.

The default training settings are:

```text
hidden layer: 16
dropout: 0.4
learning rate: 0.001
weight decay: 0.001
batch size: 64
epochs: 200
```

Train explicitly on CPU:

```bash
python scripts/train_neural.py --device cpu --run-name "cpu-run"
```

Continue from the current model weights:

```bash
python scripts/train_neural.py --resume --run-name "continued-training"
```

### 5. Generate a predictions CSV

This step is optional for the dashboard, but useful for inspecting or exporting
all predictions:

```bash
python scripts/predict_neural.py
```

Output:

```text
data/processed/round_predictions_neural.csv
```

### 6. Start the dashboard

```bash
streamlit run streamlit_app.py
```

Streamlit prints the local URL in the terminal, normally:

```text
http://localhost:8501
```

The dashboard requires both:

```text
data/processed/round_features.csv
models/round_outcome_mlp.pt
```

## Start the Dashboard with Existing Local Artifacts

If the processed dataset and trained checkpoint already exist locally, only run:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

There is no need to parse or train again.

## Parse One Demo

Parse a specific demo:

```bash
python scripts/parse_demo.py path/to/match.dem
```

If only a filename is provided and the file is not found in the current
directory, the script also checks `data/raw/demos/`.

Example:

```bash
python scripts/parse_demo.py example.dem --verbose
```

## FACEIT Demo Collection

The FACEIT helper can search player match histories, filter matches by map, and
write a manifest of available demo resources.

Set a FACEIT Data API key:

```bash
export FACEIT_DATA_API_KEY="your_api_key"
```

Build a Mirage manifest without downloading demos:

```bash
python scripts/download_faceit_demos.py \
  --players player_name \
  --map mirage \
  --history-limit 100 \
  --days-back 90 \
  --list-only
```

Search for multiple players:

```bash
python scripts/download_faceit_demos.py \
  --players player_one player_two \
  --map mirage \
  --list-only
```

Actual demo downloads depend on FACEIT download-endpoint authorization. If a
separate download token is available, set:

```bash
export FACEIT_DOWNLOADS_API_TOKEN="your_download_token"
```

Never commit API keys or tokens.

## Features

Each model row represents one round. Most comparison features are calculated as:

```text
T value - CT value
```

A positive difference therefore indicates a T-side advantage and a negative
difference indicates a CT-side advantage.

The current model uses 21 features:

### Match context

- round number
- score difference before the round
- pistol-round flag
- previous-round winner
- win-streak difference

### Economy and equipment

- CT defuse-kit count
- T buy type
- CT buy type
- buy-type difference
- total-money difference
- equipment-value difference

### Weapons and protection

- rifle-count difference
- SMG-count difference
- sniper-count difference
- armored-player difference
- helmet-player difference

### Utility

- total-utility difference
- smoke difference
- flash difference
- HE-grenade difference
- molotov/incendiary difference

The full dataset also stores historical form features calculated only from
earlier rounds. They are available for experiments but are not used by the
current default model because they did not improve held-out performance.

## Model

The current model is a small multilayer perceptron:

```text
21 normalized inputs
    -> Linear(21, 16)
    -> ReLU
    -> Dropout(0.4)
    -> Linear(16, 1)
    -> logit
    -> sigmoid
    -> T-win probability
```

Training uses:

- `BCEWithLogitsLoss` for binary classification;
- Adam optimization;
- dropout and weight decay for regularization;
- match-aware train, validation, and test splits;
- validation loss for best-model selection;
- validation accuracy for decision-threshold selection.

The saved checkpoint contains:

- learned weights and biases;
- expected feature order;
- training normalization mean and standard deviation;
- hidden-layer configuration;
- dropout value;
- decision threshold;
- held-out test metrics.

## Evaluation Design

Rounds from the same match are strongly related. Randomly splitting individual
rounds could place rounds from one match in both training and test data, producing
overly optimistic results.

This project uses `GroupShuffleSplit` with `match_id` as the group:

- complete matches stay together;
- duplicate matches are removed before splitting;
- normalization is fitted on the real training subset only;
- validation selects the best model and threshold;
- test is reserved for final evaluation.

Reported metrics include:

- accuracy;
- ROC-AUC;
- log loss;
- precision;
- recall;
- F1 score;
- confusion matrix.

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   |   |-- demos/                 Input .dem files
|   |   `-- parsed/                Raw Awpy CSV tables per demo
|   `-- processed/
|       |-- by_demo/               Full/core datasets per demo
|       |-- round_features.csv     Combined full dataset
|       |-- core_round_features.csv
|       `-- round_predictions_neural.csv
|-- experiments/
|   |-- experiment_log.jsonl       Experiment settings and scores
|   `-- models/                    Archived checkpoints per run
|-- models/
|   `-- round_outcome_mlp.pt       Current default model checkpoint
|-- scripts/
|   |-- download_faceit_demos.py
|   |-- parse_demo.py
|   |-- parse_all_demos.py
|   |-- train_neural.py
|   `-- predict_neural.py
|-- src/cs2_round_predictor/
|   |-- config.py                  Shared project paths
|   |-- datasets.py                Aggregate-dataset management
|   |-- features/core_features.py  Core feature engineering
|   |-- models/neural.py           MLP training and inference
|   `-- parsing/demo_parser.py     Awpy parsing and round snapshots
|-- streamlit_app.py               Interactive dashboard
|-- requirements.txt
`-- README.md
```

## Generated Files and Git

Raw demos, parsed data, processed datasets, and model checkpoints can be large and
are ignored by Git:

```text
data/raw/*
data/processed/*
models/
```

This means a fresh clone does not automatically contain the local dataset or
current model checkpoint. To run from a fresh clone, provide `.dem` files, parse
them, and train a model by following the Quick Start.

Experiment metadata is stored in:

```text
experiments/experiment_log.jsonl
```

## Useful Command Reference

```bash
# Show all command options
python scripts/parse_all_demos.py --help
python scripts/parse_demo.py --help
python scripts/train_neural.py --help
python scripts/predict_neural.py --help
python scripts/download_faceit_demos.py --help

# Full local pipeline
python scripts/parse_all_demos.py
python scripts/train_neural.py --run-name "baseline"
python scripts/predict_neural.py
streamlit run streamlit_app.py
```

## Troubleshooting

### `No .dem files found`

Place `.dem` files in `data/raw/demos/`, or provide another directory:

```bash
python scripts/parse_all_demos.py --input-dir path/to/demos
```

### The dashboard says the dataset is missing

Run:

```bash
python scripts/parse_all_demos.py
```

### The dashboard says the model checkpoint is missing

Run:

```bash
python scripts/train_neural.py --run-name "initial-model"
```

### Training uses CPU

CPU training is supported. CUDA is selected automatically only when PyTorch
detects an available compatible GPU. Check with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Parsing was interrupted

Run the batch parser again without `--force`. It skips demos whose expected output
paths already exist. Inspect or remove incomplete output for the interrupted demo
before retrying it.

### A checkpoint is rejected

The model loader rejects checkpoints whose feature order does not match the
current feature schema. Retrain the model after changing core features.

## Limitations

- The current dataset contains only Mirage matches.
- The application performs offline demo analysis, not live in-game prediction.
- Match-aware splitting does not guarantee that the same players never appear in
  both training and test matches.
- The dashboard shows input signals, but does not yet provide formal local
  explanations such as SHAP values.
- Missing parsed snapshot values are currently filled with zero.
- The current pipeline does not include an automated test suite.
- Model probabilities have not yet been formally calibrated.

## Future Improvements

- evaluate on a time-based external test set;
- add more maps and report performance per map;
- compare the MLP with logistic regression and gradient-boosting baselines;
- add probability-calibration metrics and reliability plots;
- add SHAP or permutation-based feature explanations;
- store exact split manifests for fully reproducible dashboard evaluation;
- add automated unit and integration tests.
