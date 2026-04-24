from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "round_number",
    "round_length_ticks",
    "bomb_planted",
    "t_kills",
    "ct_kills",
    "t_total_damage",
    "ct_total_damage",
    "t_smokes",
    "ct_smokes",
    "t_flashes",
    "ct_flashes",
    "t_he",
    "ct_he",
    "t_molotovs",
    "ct_molotovs",
    "first_kill_by_t",
    "first_kill_by_ct",
]

TARGET_COLUMN = "won_round"


def build_round_feature_table(rounds: pd.DataFrame) -> pd.DataFrame:
    dataset = rounds.loc[:, ["match_id", *FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(int)
    return dataset
