# CS2 Round Outcome Predictor

## What This Project Is

This project predicts the probability of a CS2 round being won by the T side based on structured match data extracted from CS2 demo files.

The system will:

- parse `.dem` files
- extract round-level features
- train a prediction model
- show round win probabilities and match insights in a web dashboard

## Why This Project Fits The Assignment

This project matches the assignment because it contains:

- one clear problem: predict round outcome
- one clear AI component: a trained prediction model
- one clear input/output flow
- one clear user-facing interface
- enough code and technical reasoning to explain during the oral exam

In the language of the assignment:

> input -> AI model -> useful output -> interface

For this project:

> CS2 demo data -> round prediction model -> T-side win probability and insights -> web dashboard

## Domain And Technical Path

### Main Domain

- Time-series / monitoring / prediction

Why this domain fits:

- the data comes from events over time
- the goal is to predict an outcome from structured sequential match data
- the result is well suited for a dashboard-style interface

### Main Technical Path

- Path 1: Train Your Own Neural Network

Why this path fits:

- the dataset is built from raw demo files
- the features are engineered from parsed round data
- the model is trained and evaluated by us

## Planned Pipeline

The first full pipeline is:

1. Parse one or more CS2 demo files
2. Extract round-level structured data
3. Build a dataset with one row per round
4. Train a model to predict round win/loss
5. Evaluate the model
6. Visualize predictions in a dashboard

## Input, Model, Output

### Input

- CS2 `.dem` files
- parsed round events and match information

### Example Features

- round number
- round length
- bomb planted
- kills per side
- total damage per side
- grenade usage per side
- first-kill side

### Target

- `1` = T side won the round
- `0` = CT side won the round

### Output

- predicted T-side round win probability
- actual round result
- simple match insights and visualizations

## Main Stack

### AI and Data

- Python 3.11+
- pandas
- PyTorch
- scikit-learn for baseline comparisons
- Awpy for parsing CS2 `.dem` files

### Interface

- Streamlit for a dashboard-style web interface

Possible later extension:

- FastAPI backend with a separate frontend

## Project Scope

The first version will stay focused and practical.

The project will not try to:

- join the game live
- act as a bot
- model every tick directly from the start
- become a full coaching platform in version one

The first version will focus on:

- post-match demo analysis
- round-level prediction
- a clean and explainable ML pipeline
- a simple but useful dashboard

## Recommended Milestones

### Milestone 1: Data Pipeline

- parse one demo with Awpy
- inspect available round data
- export a first CSV with one row per round

### Milestone 2: Baseline Model

- clean the dataset
- split train/test data
- train a simple baseline model
- measure performance

### Milestone 3: Neural Network

- train a small PyTorch model
- compare it to the baseline
- keep the approach that is easiest to explain and defend

### Milestone 4: Dashboard

- load processed match data
- show predicted probabilities
- compare prediction with actual result
- visualize round-by-round insights

## Best First Step

The best first practical step is not model training.

The best first step is:

1. install the project requirements
2. parse one demo file
3. inspect what data exists per round
4. save a first round-level dataset

Only after that should model training begin.

## Awpy Notes

The project uses Awpy as the primary parser for CS2 demos.

According to the Awpy documentation:

- parsing starts with `from awpy import Demo`
- you run `dem = Demo(path_to_demo)` and then `dem.parse()`
- Awpy exposes parsed tables such as `rounds`, `kills`, `damages`, `grenades`, `bomb`, and `ticks`
- Awpy 2.0.2 requires Python 3.11 or newer

References:

- https://awpy.readthedocs.io/en/latest/examples/parse_demo.html
- https://awpy.readthedocs.io/en/latest/modules/parser_output.html

## Short Project Pitch

CS2 Round Outcome Predictor is a CS2 analytics application that predicts the probability of a round being won by the T side from demo-derived match data. The project builds a full AI pipeline from raw `.dem` files to structured round features, model training, evaluation, and a dashboard for match analysis. The goal is not to build a game bot, but a useful AI-assisted analysis tool that is technically clear, demoable, and explainable.
