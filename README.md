# CS2 Round Outcome Predictor

## What This Project Is

This project predicts the probability of a CS2 round being won by the T side based on structured pre-round match data extracted from CS2 demo files.

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

- map name
- round number
- pistol round yes/no
- total money per side at round start
- equipment value per side at freeze-end
- rifles / SMGs / snipers per side
- armor and helmets per side
- defuse kits for CT
- utility count per side
- previous round winner
- T-side win streak
- CT-side win streak

### Target

- `1` = T side won the round
- `0` = CT side won the round

### First Dataset Schema

The first training dataset uses one row per round with these columns:

- `match_id`
- `map_name`
- `round_number`
- `is_pistol_round`
- `t_money_total`
- `ct_money_total`
- `t_equipment_value`
- `ct_equipment_value`
- `t_rifles`
- `ct_rifles`
- `t_smgs`
- `ct_smgs`
- `t_snipers`
- `ct_snipers`
- `t_armor_players`
- `ct_armor_players`
- `t_helmet_players`
- `ct_helmet_players`
- `ct_defuse_kits`
- `t_utility_total`
- `ct_utility_total`
- `t_smokes`
- `ct_smokes`
- `t_flashes`
- `ct_flashes`
- `t_he`
- `ct_he`
- `t_molotovs`
- `ct_molotovs`
- `previous_round_winner`
- `t_win_streak`
- `ct_win_streak`
- `won_round`

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
- pre-round round-level prediction
- a clean and explainable ML pipeline
- a simple but useful dashboard

## Recommended Milestones

### Milestone 1: Data Pipeline

- parse one demo with Awpy
- take one freeze-end snapshot per round
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
3. inspect freeze-end player state per round
4. save a first pre-round dataset

Only after that should model training begin.

## Awpy Notes

The project uses Awpy as the primary parser for CS2 demos.

According to the Awpy documentation:

- parsing starts with `from awpy import Demo`
- you run `dem = Demo(path_to_demo)` and then `dem.parse()`
- Awpy exposes parsed tables such as `rounds`, `kills`, `damages`, `grenades`, `bomb`, and `ticks`
- the `ticks` table can include player properties such as `balance`, `start_balance`, `current_equip_value`, `armor_value`, `has_helmet`, `has_defuser`, and `inventory`
- Awpy 2.0.2 requires Python 3.11 or newer

References:

- https://awpy.readthedocs.io/en/latest/examples/parse_demo.html
- https://awpy.readthedocs.io/en/latest/modules/parser_output.html

## Short Project Pitch

CS2 Round Outcome Predictor is a CS2 analytics application that predicts the probability of a round being won by the T side from demo-derived pre-round match data. The project builds a full AI pipeline from raw `.dem` files to structured round features, model training, evaluation, and a dashboard for match analysis. The goal is not to build a game bot, but a useful AI-assisted analysis tool that is technically clear, demoable, and explainable.
