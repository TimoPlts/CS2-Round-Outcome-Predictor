from __future__ import annotations

import pandas as pd


PAIR_DIFFS = {
    "money_total_diff": ("t_money_total", "ct_money_total"),
    "equipment_value_diff": ("t_equipment_value", "ct_equipment_value"),
    "rifles_diff": ("t_rifles", "ct_rifles"),
    "smgs_diff": ("t_smgs", "ct_smgs"),
    "snipers_diff": ("t_snipers", "ct_snipers"),
    "armor_players_diff": ("t_armor_players", "ct_armor_players"),
    "helmet_players_diff": ("t_helmet_players", "ct_helmet_players"),
    "utility_total_diff": ("t_utility_total", "ct_utility_total"),
    "smokes_diff": ("t_smokes", "ct_smokes"),
    "flashes_diff": ("t_flashes", "ct_flashes"),
    "he_diff": ("t_he", "ct_he"),
    "molotovs_diff": ("t_molotovs", "ct_molotovs"),
}

CORE_FEATURE_COLUMNS = [
    "round_number",
    "score_diff",
    "is_pistol_round",
    "previous_round_winner",
    "win_streak_diff",
    "ct_defuse_kits",
    "t_buy_type",
    "ct_buy_type",
    "buy_type_diff",
    *PAIR_DIFFS.keys(),
]

MATCH_ID_COLUMN = "match_id"
TARGET_COLUMN = "won_round"


def build_core_feature_table(dataset: pd.DataFrame) -> pd.DataFrame:
    core = dataset.loc[:, [MATCH_ID_COLUMN]].copy()
    core["round_number"] = dataset["round_number"].astype(int)
    previous_t_wins = dataset.groupby(MATCH_ID_COLUMN, sort=False)[TARGET_COLUMN].cumsum()
    previous_t_wins = previous_t_wins - dataset[TARGET_COLUMN].astype(int)
    previous_rounds = dataset.groupby(MATCH_ID_COLUMN, sort=False).cumcount()
    previous_ct_wins = previous_rounds - previous_t_wins
    core["score_diff"] = previous_t_wins - previous_ct_wins
    core["is_pistol_round"] = dataset["is_pistol_round"].astype(int)
    core["previous_round_winner"] = dataset["previous_round_winner"].astype(int)
    core["win_streak_diff"] = dataset["t_win_streak"] - dataset["ct_win_streak"]
    core["ct_defuse_kits"] = dataset["ct_defuse_kits"].astype(int)
    core["t_buy_type"] = _classify_buy_type(dataset["t_equipment_value"])
    core["ct_buy_type"] = _classify_buy_type(dataset["ct_equipment_value"])
    core["buy_type_diff"] = core["t_buy_type"] - core["ct_buy_type"]

    for feature_name, (t_column, ct_column) in PAIR_DIFFS.items():
        core[feature_name] = dataset[t_column] - dataset[ct_column]

    core[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(int)
    return core.loc[:, [MATCH_ID_COLUMN, *CORE_FEATURE_COLUMNS, TARGET_COLUMN]]


def _classify_buy_type(equipment_value: pd.Series) -> pd.Series:
    values = equipment_value.astype(float)
    return pd.cut(
        values,
        bins=[float("-inf"), 10_000, 20_000, float("inf")],
        labels=[0, 1, 2],
        right=False,
    ).astype(int)
