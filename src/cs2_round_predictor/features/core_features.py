from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PAIR_DIFFS = {
    "money_total_diff": ("t_money_total", "ct_money_total"),
    "equipment_value_diff": ("t_equipment_value", "ct_equipment_value"),
    "armor_players_diff": ("t_armor_players", "ct_armor_players"),
    "helmet_players_diff": ("t_helmet_players", "ct_helmet_players"),
    "utility_total_diff": ("t_utility_total", "ct_utility_total"),
    "smokes_diff": ("t_smokes", "ct_smokes"),
    "flashes_diff": ("t_flashes", "ct_flashes"),
}

CORE_FEATURE_COLUMNS = [
    "is_pistol_round",
    "previous_round_winner",
    "win_streak_diff",
    "ct_defuse_kits",
    *PAIR_DIFFS.keys(),
]

TARGET_COLUMN = "won_round"


def build_core_feature_table(dataset: pd.DataFrame) -> pd.DataFrame:
    core = dataset.loc[:, ["match_id"]].copy()
    core["is_pistol_round"] = dataset["is_pistol_round"].astype(int)
    core["previous_round_winner"] = dataset["previous_round_winner"].astype(int)
    core["win_streak_diff"] = dataset["t_win_streak"] - dataset["ct_win_streak"]
    core["ct_defuse_kits"] = dataset["ct_defuse_kits"].astype(int)

    for feature_name, (t_column, ct_column) in PAIR_DIFFS.items():
        core[feature_name] = dataset[t_column] - dataset[ct_column]

    core[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(int)
    return core.loc[:, ["match_id", *CORE_FEATURE_COLUMNS, TARGET_COLUMN]]


def rank_core_features(core_dataset: pd.DataFrame) -> list[tuple[str, float]]:
    X = core_dataset.loc[:, CORE_FEATURE_COLUMNS]
    y = core_dataset[TARGET_COLUMN].astype(int)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )
    model.fit(X, y)

    coefficients = model.named_steps["classifier"].coef_[0]
    ranking = sorted(
        (
            (feature_name, float(abs(coefficient)))
            for feature_name, coefficient in zip(CORE_FEATURE_COLUMNS, coefficients)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranking
