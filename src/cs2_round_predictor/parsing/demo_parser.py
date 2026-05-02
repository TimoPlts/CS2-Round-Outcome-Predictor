from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ast
import json
import re

import pandas as pd


EXPECTED_COLUMNS = {
    "match_id",
    "map_name",
    "round_number",
    "is_pistol_round",
    "t_money_total",
    "ct_money_total",
    "t_equipment_value",
    "ct_equipment_value",
    "t_rifles",
    "ct_rifles",
    "t_smgs",
    "ct_smgs",
    "t_snipers",
    "ct_snipers",
    "t_armor_players",
    "ct_armor_players",
    "t_helmet_players",
    "ct_helmet_players",
    "ct_defuse_kits",
    "t_utility_total",
    "ct_utility_total",
    "t_smokes",
    "ct_smokes",
    "t_flashes",
    "ct_flashes",
    "t_he",
    "ct_he",
    "t_molotovs",
    "ct_molotovs",
    "previous_round_winner",
    "t_win_streak",
    "ct_win_streak",
    "won_round",
}

SNAPSHOT_DEFAULTS = {
    "t_money_total": 0,
    "ct_money_total": 0,
    "t_equipment_value": 0,
    "ct_equipment_value": 0,
    "t_rifles": 0,
    "ct_rifles": 0,
    "t_smgs": 0,
    "ct_smgs": 0,
    "t_snipers": 0,
    "ct_snipers": 0,
    "t_armor_players": 0,
    "ct_armor_players": 0,
    "t_helmet_players": 0,
    "ct_helmet_players": 0,
    "ct_defuse_kits": 0,
    "t_utility_total": 0,
    "ct_utility_total": 0,
    "t_smokes": 0,
    "ct_smokes": 0,
    "t_flashes": 0,
    "ct_flashes": 0,
    "t_he": 0,
    "ct_he": 0,
    "t_molotovs": 0,
    "ct_molotovs": 0,
}

RIFLE_ITEMS = {
    "ak47",
    "aug",
    "famas",
    "galilar",
    "m4a1",
    "m4a1_silencer",
    "sg556",
    "sg553",
}
SMG_ITEMS = {
    "bizon",
    "mac10",
    "mp5sd",
    "mp7",
    "mp9",
    "p90",
    "ump45",
}
SNIPER_ITEMS = {
    "awp",
    "g3sg1",
    "scar20",
    "ssg08",
}
UTILITY_ALIASES = {
    "flashbang": "flashbang",
    "hegrenade": "hegrenade",
    "molotov": "molotov",
    "incgrenade": "molotov",
    "incendiarygrenade": "molotov",
    "smokegrenade": "smokegrenade",
    "decoy": "decoy",
}
IGNORED_INVENTORY_ITEMS = {
    "c4",
    "defuser",
    "knife",
    "knife_t",
    "bayonet",
    "fists",
    "taser",
    "weapon",
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


def parse_demo_file(
    demo_path: str | Path,
    *,
    verbose: bool = False,
) -> ParsedDemoArtifacts:
    try:
        from awpy import Demo
    except ImportError as exc:
        raise ImportError(
            "Awpy is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    path = Path(demo_path)
    demo = Demo(path, verbose=verbose)
    demo.parse(
        player_props=[
            "armor_value",
            "balance",
            "cash_spent_this_round",
            "current_equip_value",
            "has_defuser",
            "has_helmet",
            "inventory",
            "start_balance",
        ]
    )
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
    artifacts: ParsedDemoArtifacts,
    *,
    match_id: str | None = None,
) -> pd.DataFrame:
    rounds = _normalize_rounds(artifacts.rounds, artifacts.header, match_id)
    snapshot = _build_preround_snapshot(rounds, artifacts.ticks)
    dataset = rounds.merge(snapshot, on="round_number", how="left").fillna(SNAPSHOT_DEFAULTS)

    ordered_columns = [
        "match_id",
        "map_name",
        "round_number",
        "is_pistol_round",
        "t_money_total",
        "ct_money_total",
        "t_equipment_value",
        "ct_equipment_value",
        "t_rifles",
        "ct_rifles",
        "t_smgs",
        "ct_smgs",
        "t_snipers",
        "ct_snipers",
        "t_armor_players",
        "ct_armor_players",
        "t_helmet_players",
        "ct_helmet_players",
        "ct_defuse_kits",
        "t_utility_total",
        "ct_utility_total",
        "t_smokes",
        "ct_smokes",
        "t_flashes",
        "ct_flashes",
        "t_he",
        "ct_he",
        "t_molotovs",
        "ct_molotovs",
        "previous_round_winner",
        "t_win_streak",
        "ct_win_streak",
        "won_round",
    ]

    for column in ordered_columns:
        if column not in dataset.columns:
            dataset[column] = 0

    int_columns = [column for column in ordered_columns if column not in {"match_id", "map_name"}]
    dataset[int_columns] = dataset[int_columns].astype(int)
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
    rounds: pd.DataFrame,
    header: dict[str, Any],
    match_id: str | None,
) -> pd.DataFrame:
    df = rounds.rename(columns={"round_num": "round_number"}).copy()
    df = df.sort_values("round_number").reset_index(drop=True)

    map_name = _normalize_map_name(header.get("map_name")) or "unknown_map"
    df["match_id"] = match_id or map_name
    df["map_name"] = map_name
    df["won_round"] = (df["winner"].astype(str).str.upper() == "T").astype(int)
    df["is_pistol_round"] = df["round_number"].isin({1, 13}).astype(int)

    previous_winner = df["won_round"].shift(1)
    df["previous_round_winner"] = previous_winner.fillna(-1).astype(int)

    t_streak = 0
    ct_streak = 0
    t_streak_values: list[int] = []
    ct_streak_values: list[int] = []
    for winner in df["won_round"]:
        t_streak_values.append(t_streak)
        ct_streak_values.append(ct_streak)
        if winner == 1:
            t_streak += 1
            ct_streak = 0
        else:
            ct_streak += 1
            t_streak = 0

    df["t_win_streak"] = t_streak_values
    df["ct_win_streak"] = ct_streak_values
    return df.loc[:, [
        "match_id",
        "map_name",
        "round_number",
        "freeze_end",
        "is_pistol_round",
        "previous_round_winner",
        "t_win_streak",
        "ct_win_streak",
        "won_round",
    ]]


def _build_preround_snapshot(rounds: pd.DataFrame, ticks: pd.DataFrame) -> pd.DataFrame:
    if ticks.empty:
        return pd.DataFrame(columns=["round_number", *SNAPSHOT_DEFAULTS])

    ticks_df = ticks.rename(columns={"round_num": "round_number"}).copy()
    if "round_number" not in ticks_df.columns or "tick" not in ticks_df.columns:
        raise ValueError("Awpy ticks data must contain `round_num`/`round_number` and `tick`.")

    side_column = _resolve_column(ticks_df, ["side"])
    if side_column is None:
        raise ValueError("Awpy ticks data must contain a `side` column for pre-round snapshots.")

    id_column = _resolve_column(ticks_df, ["steamid", "player_steamid", "steam_id", "name"])
    if id_column is None:
        raise ValueError("Awpy ticks data must contain a player identifier column.")

    ticks_df = ticks_df.merge(
        rounds.loc[:, ["round_number", "freeze_end"]],
        on="round_number",
        how="inner",
    )
    ticks_df["side"] = ticks_df[side_column].astype(str).str.lower()
    ticks_df["actor_id"] = ticks_df[id_column].astype(str)
    ticks_df = ticks_df[ticks_df["side"].isin(["t", "ct"])].copy()
    if ticks_df.empty:
        return pd.DataFrame(columns=["round_number", *SNAPSHOT_DEFAULTS])

    eligible = ticks_df[ticks_df["tick"] >= ticks_df["freeze_end"]].copy()
    if eligible.empty:
        eligible = ticks_df.copy()

    snapshot_ticks = (
        eligible.groupby("round_number")["tick"]
        .min()
        .rename("snapshot_tick")
        .reset_index()
    )
    snapshot = ticks_df.merge(snapshot_ticks, on="round_number", how="inner")
    snapshot = snapshot[snapshot["tick"] == snapshot["snapshot_tick"]].copy()
    snapshot = snapshot.drop_duplicates(subset=["round_number", "actor_id", "side"])

    snapshot["money_total"] = _numeric_series(snapshot, ["start_balance", "balance"])
    snapshot["equipment_value"] = _numeric_series(
        snapshot,
        ["current_equip_value", "round_start_equip_value"],
    )
    snapshot["armor_player"] = (_numeric_series(snapshot, ["armor_value", "armor"]) > 0).astype(int)
    snapshot["helmet_player"] = _bool_series(snapshot, ["has_helmet"])
    snapshot["defuse_kit"] = _bool_series(snapshot, ["has_defuser"])

    inventory_names = snapshot["inventory"].apply(_extract_inventory_names) if "inventory" in snapshot.columns else pd.Series([[]] * len(snapshot), index=snapshot.index)
    inventory_summary = inventory_names.apply(_summarize_inventory)
    inventory_df = pd.DataFrame(inventory_summary.tolist(), index=snapshot.index)
    snapshot = pd.concat([snapshot, inventory_df], axis=1)

    summary = (
        snapshot.groupby(["round_number", "side"], as_index=False)[
            [
                "money_total",
                "equipment_value",
                "rifle_player",
                "smg_player",
                "sniper_player",
                "armor_player",
                "helmet_player",
                "defuse_kit",
                "utility_total",
                "smokes",
                "flashes",
                "he",
                "molotovs",
            ]
        ]
        .sum()
    )

    output = pd.DataFrame({"round_number": rounds["round_number"]})
    output = output.merge(
        _pivot_side_metric(summary, "money_total", "t_money_total", "ct_money_total"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "equipment_value", "t_equipment_value", "ct_equipment_value"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "rifle_player", "t_rifles", "ct_rifles"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "smg_player", "t_smgs", "ct_smgs"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "sniper_player", "t_snipers", "ct_snipers"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "armor_player", "t_armor_players", "ct_armor_players"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "helmet_player", "t_helmet_players", "ct_helmet_players"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "defuse_kit", "t_unused_defuse_kits", "ct_defuse_kits"),
        on="round_number",
        how="left",
    ).drop(columns=["t_unused_defuse_kits"])
    output = output.merge(
        _pivot_side_metric(summary, "utility_total", "t_utility_total", "ct_utility_total"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "smokes", "t_smokes", "ct_smokes"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "flashes", "t_flashes", "ct_flashes"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "he", "t_he", "ct_he"),
        on="round_number",
        how="left",
    )
    output = output.merge(
        _pivot_side_metric(summary, "molotovs", "t_molotovs", "ct_molotovs"),
        on="round_number",
        how="left",
    )
    return output.fillna(SNAPSHOT_DEFAULTS)


def _pivot_side_metric(
    summary: pd.DataFrame,
    metric: str,
    t_name: str,
    ct_name: str,
) -> pd.DataFrame:
    pivoted = (
        summary.pivot(index="round_number", columns="side", values=metric)
        .reset_index()
        .rename(columns={"t": t_name, "ct": ct_name})
    )
    for column in [t_name, ct_name]:
        if column not in pivoted.columns:
            pivoted[column] = 0
    return pivoted.loc[:, ["round_number", t_name, ct_name]]


def _numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column = _resolve_column(df, candidates)
    if column is None:
        return pd.Series(0, index=df.index, dtype="int64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def _bool_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column = _resolve_column(df, candidates)
    if column is None:
        return pd.Series(0, index=df.index, dtype="int64")
    series = df[column]
    if series.dtype == bool:
        return series.astype(int)
    normalized = series.astype(str).str.lower()
    return normalized.isin({"1", "true", "t", "yes"}).astype(int)


def _normalize_map_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return normalized.removeprefix("de_")


def _extract_inventory_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        return _extract_inventory_names(value.tolist())
    if isinstance(value, (list, tuple, set)):
        extracted: list[str] = []
        for item in value:
            extracted.extend(_extract_inventory_names(item))
        return extracted
    if isinstance(value, dict):
        for key in ["weapon_name", "name", "item_name", "item"]:
            if key in value:
                return _extract_inventory_names(value[key])
        extracted: list[str] = []
        for nested in value.values():
            extracted.extend(_extract_inventory_names(nested))
        return extracted
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
            return []
        if stripped[0] in "[{":
            for loader in (json.loads, ast.literal_eval):
                try:
                    return _extract_inventory_names(loader(stripped))
                except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                    continue
        quoted_tokens = re.findall(r'"([^"]+)"|\'([^\']+)\'', stripped)
        if quoted_tokens:
            names: list[str] = []
            for left, right in quoted_tokens:
                names.extend(_extract_inventory_names(left or right))
            normalized = [_normalize_item_name(name) for name in names]
            return [name for name in normalized if name]

        normalized = [_normalize_item_name(token) for token in re.split(r"[^a-zA-Z0-9_]+", stripped)]
        return [name for name in normalized if name]
    return []


def _normalize_item_name(value: str) -> str | None:
    normalized = value.strip().lower()
    if not normalized:
        return None
    normalized = normalized.removeprefix("weapon_")
    normalized = normalized.removeprefix("item_")
    normalized = normalized.replace("-", "").replace(" ", "").replace(".", "")
    alias_map = {
        "flash": "flashbang",
        "he": "hegrenade",
        "incendiary": "incgrenade",
        "smoke": "smokegrenade",
        "m4a4": "m4a1",
        "sg55": "sg553",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized in IGNORED_INVENTORY_ITEMS:
        return None
    if normalized in RIFLE_ITEMS | SMG_ITEMS | SNIPER_ITEMS | set(UTILITY_ALIASES):
        return normalized
    return None


def _summarize_inventory(items: list[str]) -> dict[str, int]:
    utility_counts = {
        "smokes": 0,
        "flashes": 0,
        "he": 0,
        "molotovs": 0,
        "utility_total": 0,
    }

    normalized_items = [_normalize_item_name(item) for item in items]
    normalized_items = [item for item in normalized_items if item]

    for item in normalized_items:
        utility_type = UTILITY_ALIASES.get(item)
        if utility_type == "smokegrenade":
            utility_counts["smokes"] += 1
            utility_counts["utility_total"] += 1
        elif utility_type == "flashbang":
            utility_counts["flashes"] += 1
            utility_counts["utility_total"] += 1
        elif utility_type == "hegrenade":
            utility_counts["he"] += 1
            utility_counts["utility_total"] += 1
        elif utility_type == "molotov":
            utility_counts["molotovs"] += 1
            utility_counts["utility_total"] += 1
        elif utility_type == "decoy":
            utility_counts["utility_total"] += 1

    return {
        "rifle_player": int(any(item in RIFLE_ITEMS for item in normalized_items)),
        "smg_player": int(any(item in SMG_ITEMS for item in normalized_items)),
        "sniper_player": int(any(item in SNIPER_ITEMS for item in normalized_items)),
        **utility_counts,
    }


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None
