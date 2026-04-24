from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_COLUMNS = {
    "match_id",
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
    "won_round",
}


@dataclass(slots=True)
class ParsedRounds:
    rounds: pd.DataFrame


@dataclass(slots=True)
class ParsedDemoArtifacts:
    header: dict[str, Any]
    rounds: pd.DataFrame
    kills: pd.DataFrame
    damages: pd.DataFrame
    grenades: pd.DataFrame
    bomb: pd.DataFrame
    ticks: pd.DataFrame


def load_rounds_csv(csv_path: str | Path) -> ParsedRounds:
    path = Path(csv_path)
    rounds = pd.read_csv(path)
    missing = sorted(EXPECTED_COLUMNS.difference(rounds.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {', '.join(missing)}"
        )
    return ParsedRounds(rounds=rounds)


def parse_demo_file(demo_path: str | Path, *, verbose: bool = False) -> ParsedDemoArtifacts:
    try:
        from awpy import Demo
    except ImportError as exc:
        raise ImportError(
            "Awpy is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    path = Path(demo_path)
    demo = Demo(path, verbose=verbose)
    demo.parse()
    return ParsedDemoArtifacts(
        header=demo.header,
        rounds=demo.rounds.to_pandas(),
        kills=demo.kills.to_pandas(),
        damages=demo.damages.to_pandas(),
        grenades=demo.grenades.to_pandas(),
        bomb=demo.bomb.to_pandas(),
        ticks=demo.ticks.to_pandas(),
    )


def build_round_dataset_from_artifacts(
    artifacts: ParsedDemoArtifacts, *, match_id: str | None = None
) -> pd.DataFrame:
    rounds = _normalize_rounds(artifacts.rounds, artifacts.header, match_id)
    side_lookup = _build_side_lookup(artifacts.ticks)
    kills = _prepare_event_with_round_and_side(
        artifacts.kills,
        rounds,
        side_lookup,
        actor_id_candidates=["attacker_steamid", "attacker_steam_id", "attacker_name"],
    )
    damages = _prepare_event_with_round_and_side(
        artifacts.damages,
        rounds,
        side_lookup,
        actor_id_candidates=["attacker_steamid", "attacker_steam_id", "attacker_name"],
    )
    grenades = _prepare_event_with_round_and_side(
        artifacts.grenades,
        rounds,
        side_lookup,
        actor_id_candidates=["thrower_steamid", "thrower"],
    )

    kill_summary = _summarize_kills(kills)
    damage_summary = _summarize_damages(damages)
    grenade_summary = _summarize_grenades(grenades)

    dataset = (
        rounds.merge(kill_summary, on="round_number", how="left")
        .merge(damage_summary, on="round_number", how="left")
        .merge(grenade_summary, on="round_number", how="left")
        .fillna(
            {
                "bomb_planted": 0,
                "t_kills": 0,
                "ct_kills": 0,
                "t_total_damage": 0,
                "ct_total_damage": 0,
                "t_smokes": 0,
                "ct_smokes": 0,
                "t_flashes": 0,
                "ct_flashes": 0,
                "t_he": 0,
                "ct_he": 0,
                "t_molotovs": 0,
                "ct_molotovs": 0,
                "first_kill_by_t": 0,
                "first_kill_by_ct": 0,
            }
        )
    )

    ordered_columns = [
        "match_id",
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
        "won_round",
    ]
    return dataset.loc[:, ordered_columns]


def export_artifacts_to_csv(artifacts: ParsedDemoArtifacts, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([artifacts.header]).to_csv(output_path / "header.csv", index=False)
    artifacts.rounds.to_csv(output_path / "rounds.csv", index=False)
    artifacts.kills.to_csv(output_path / "kills.csv", index=False)
    artifacts.damages.to_csv(output_path / "damages.csv", index=False)
    artifacts.grenades.to_csv(output_path / "grenades.csv", index=False)
    artifacts.bomb.to_csv(output_path / "bomb.csv", index=False)
    artifacts.ticks.to_csv(output_path / "ticks.csv", index=False)


def _normalize_rounds(
    rounds: pd.DataFrame, header: dict[str, Any], match_id: str | None
) -> pd.DataFrame:
    df = rounds.rename(columns={"round_num": "round_number"}).copy()
    df["match_id"] = match_id or header.get("map_name") or "unknown_match"
    df["round_length_ticks"] = (df["end"] - df["freeze_end"]).clip(lower=0)
    df["bomb_planted"] = df["bomb_plant"].notna().astype(int)
    df["won_round"] = (df["winner"].astype(str).str.upper() == "T").astype(int)
    return df.loc[:, [
        "match_id",
        "round_number",
        "start",
        "end",
        "round_length_ticks",
        "bomb_planted",
        "won_round",
    ]]


def _build_side_lookup(ticks: pd.DataFrame) -> pd.DataFrame:
    ticks_df = ticks.rename(columns={"round_num": "round_number"}).copy()
    id_col = _resolve_column(ticks_df, ["steamid", "player_steamid", "steam_id", "name"])
    if id_col is None or "side" not in ticks_df.columns:
        raise ValueError("Awpy ticks data must contain a player identifier and `side`.")
    lookup = ticks_df.loc[:, ["round_number", id_col, "side"]].copy()
    lookup = lookup.rename(columns={id_col: "actor_id"})
    lookup["actor_id"] = lookup["actor_id"].astype(str)
    lookup["side"] = lookup["side"].astype(str).str.lower()
    return lookup.drop_duplicates(subset=["round_number", "actor_id", "side"])


def _prepare_event_with_round_and_side(
    events: pd.DataFrame,
    rounds: pd.DataFrame,
    side_lookup: pd.DataFrame,
    *,
    actor_id_candidates: list[str],
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["round_number", "tick", "side"])

    df = events.copy()
    if "round_num" in df.columns and "round_number" not in df.columns:
        df = df.rename(columns={"round_num": "round_number"})

    if "round_number" not in df.columns:
        df = _attach_round_number_from_tick(df, rounds)

    actor_id_col = _resolve_column(df, actor_id_candidates)
    if actor_id_col is not None:
        df["actor_id"] = df[actor_id_col].astype(str)
        df = df.merge(side_lookup, on=["round_number", "actor_id"], how="left")
    else:
        df["side"] = None

    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.lower()
    return df


def _attach_round_number_from_tick(events: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    if "tick" not in events.columns:
        raise ValueError("Event dataframe needs either `round_num` or `tick`.")

    round_windows = rounds.loc[:, ["round_number", "start", "end"]].sort_values("start")
    event_df = events.sort_values("tick").copy()
    attached = pd.merge_asof(
        event_df,
        round_windows,
        left_on="tick",
        right_on="start",
        direction="backward",
    )
    attached = attached[attached["tick"] <= attached["end"]].copy()
    return attached.drop(columns=["start", "end"])


def _summarize_kills(kills: pd.DataFrame) -> pd.DataFrame:
    if kills.empty:
        return pd.DataFrame(columns=["round_number", "t_kills", "ct_kills", "first_kill_by_t", "first_kill_by_ct"])

    grouped = kills.groupby("round_number")
    summary = grouped["side"].value_counts().unstack(fill_value=0).reset_index()
    summary = summary.rename(columns={"t": "t_kills", "ct": "ct_kills"})
    for col in ["t_kills", "ct_kills"]:
        if col not in summary.columns:
            summary[col] = 0

    first_kills = kills.sort_values("tick").groupby("round_number").first().reset_index()
    first_kills["first_kill_by_t"] = (first_kills["side"] == "t").astype(int)
    first_kills["first_kill_by_ct"] = (first_kills["side"] == "ct").astype(int)
    return summary.merge(
        first_kills.loc[:, ["round_number", "first_kill_by_t", "first_kill_by_ct"]],
        on="round_number",
        how="left",
    )


def _summarize_damages(damages: pd.DataFrame) -> pd.DataFrame:
    if damages.empty:
        return pd.DataFrame(columns=["round_number", "t_total_damage", "ct_total_damage"])

    damage_col = _resolve_column(damages, ["dmg_health_real", "dmg_health"])
    if damage_col is None:
        raise ValueError("Awpy damages dataframe does not contain a damage column.")

    summary = (
        damages.groupby(["round_number", "side"])[damage_col]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"t": "t_total_damage", "ct": "ct_total_damage"})
    )
    for col in ["t_total_damage", "ct_total_damage"]:
        if col not in summary.columns:
            summary[col] = 0
    return summary


def _summarize_grenades(grenades: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "t_smokes",
        "ct_smokes",
        "t_flashes",
        "ct_flashes",
        "t_he",
        "ct_he",
        "t_molotovs",
        "ct_molotovs",
    ]
    if grenades.empty:
        return pd.DataFrame(columns=["round_number", *expected])

    grenade_map = {
        ("t", "smoke"): "t_smokes",
        ("ct", "smoke"): "ct_smokes",
        ("t", "flashbang"): "t_flashes",
        ("ct", "flashbang"): "ct_flashes",
        ("t", "he"): "t_he",
        ("ct", "he"): "ct_he",
        ("t", "molotov"): "t_molotovs",
        ("ct", "molotov"): "ct_molotovs",
        ("t", "incendiary"): "t_molotovs",
        ("ct", "incendiary"): "ct_molotovs",
    }

    df = grenades.copy()
    df["grenade_type"] = df["grenade_type"].astype(str).str.lower()
    df["feature_name"] = df.apply(
        lambda row: grenade_map.get((row.get("side"), row.get("grenade_type"))),
        axis=1,
    )
    df = df[df["feature_name"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["round_number", *expected])

    summary = (
        df.groupby(["round_number", "feature_name"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in expected:
        if col not in summary.columns:
            summary[col] = 0
    return summary


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None
