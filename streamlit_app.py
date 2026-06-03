from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from cs2_round_predictor.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_EXPERIMENT_LOG_PATH,
    DEFAULT_NEURAL_MODEL_PATH,
)
from cs2_round_predictor.features.core_features import CORE_FEATURE_COLUMNS, build_core_feature_table
from cs2_round_predictor.models.neural import (
    load_neural_checkpoint,
    predict_round_probabilities_neural,
)


GROUP_TEST_SIZE = 0.25
SPLIT_RANDOM_STATE = 42
MAX_SPLIT_ATTEMPTS = 25


FEATURE_HELP = {
    "is_pistol_round": "Shows whether this round is a pistol round.",
    "previous_round_winner": "Shows who won the previous round: T, CT, or none if this is round 1.",
    "win_streak_diff": "T-side win streak minus CT-side win streak before this round starts.",
    "ct_defuse_kits": "How many CT players have a defuse kit at round start.",
    "money_total_diff": "Difference in total team money at round start: T minus CT.",
    "equipment_value_diff": "Difference in equipment value at round start: T minus CT.",
    "armor_players_diff": "Difference in number of armored players: T minus CT.",
    "helmet_players_diff": "Difference in number of players with helmets: T minus CT.",
    "utility_total_diff": "Difference in total utility items available: T minus CT.",
    "smokes_diff": "Difference in number of smoke grenades: T minus CT.",
    "flashes_diff": "Difference in number of flashbangs: T minus CT.",
}

st.set_page_config(
    page_title="CS2 Round Outcome Predictor",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4efe4;
            --paper: rgba(255, 252, 245, 0.92);
            --ink: #1d1f21;
            --muted: #66624f;
            --tint-t: #e07a5f;
            --tint-ct: #3d5a80;
            --accent: #81b29a;
            --border: rgba(29, 31, 33, 0.12);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(224, 122, 95, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(61, 90, 128, 0.18), transparent 28%),
                linear-gradient(180deg, #f9f5eb 0%, #efe7d6 100%);
            color: var(--ink);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            font-family: "Trebuchet MS", "Gill Sans", sans-serif;
            letter-spacing: 0.02em;
        }
        .hero-card, .panel-card, .outcome-card {
            background: var(--paper);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: 0 18px 40px rgba(40, 28, 10, 0.08);
        }
        .hero-card {
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.2rem;
        }
        .panel-card {
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.8rem;
        }
        .pill {
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(29, 31, 33, 0.08);
            font-size: 0.85rem;
        }
        .outcome-card {
            padding: 1rem 1.1rem;
        }
        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 16px;
            border: 1px solid rgba(29, 31, 33, 0.12);
            background: rgba(255, 252, 245, 0.96);
            color: var(--ink);
            font-weight: 700;
            box-shadow: 0 10px 24px rgba(40, 28, 10, 0.08);
        }
        div.stButton > button:hover {
            border-color: rgba(129, 178, 154, 0.9);
            color: var(--ink);
            background: #fffdf8;
        }
        div.stButton > button:disabled {
            background: rgba(240, 235, 225, 0.92);
            color: rgba(29, 31, 33, 0.38);
            border-color: rgba(29, 31, 33, 0.08);
            box-shadow: none;
        }
        .signal {
            font-size: 2.6rem;
            line-height: 1;
            font-weight: 800;
            margin-top: 0.25rem;
        }
        .signal.t {
            color: var(--tint-t);
        }
        .signal.ct {
            color: var(--tint-ct);
        }
        .subtle {
            color: var(--muted);
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_dashboard_frame(dataset_path: str, model_path: str) -> pd.DataFrame:
    round_df = pd.read_csv(dataset_path).reset_index(drop=True)
    core_df = build_core_feature_table(round_df)
    predictions = predict_round_probabilities_neural(core_df, model_path=model_path)

    dashboard_df = round_df.copy()
    for column in core_df.columns:
        if column not in dashboard_df.columns:
            dashboard_df[column] = core_df[column]
    dashboard_df["t_win_probability"] = predictions["t_win_probability"]
    dashboard_df["predicted_won_round"] = predictions["predicted_won_round"]
    dashboard_df["predicted_winner"] = predictions["predicted_winner"]
    dashboard_df["actual_winner"] = dashboard_df["won_round"].map({1: "T", 0: "CT"})
    dashboard_df["model_correct"] = (
        dashboard_df["predicted_won_round"] == dashboard_df["won_round"]
    )
    dashboard_df["confidence_pct"] = (dashboard_df["t_win_probability"] * 100).round(1)
    return dashboard_df


def _render_core_feature_table(row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="section-title">Core Features</div>
            <div class="subtle">These are the exact features that go into the neural network for this round.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature_rows = []
    for feature_name in CORE_FEATURE_COLUMNS:
        raw_value = row[feature_name]
        value = int(raw_value) if float(raw_value).is_integer() else round(float(raw_value), 3)
        feature_rows.append(
            {
                "Feature": feature_name,
                "Value": value,
                "Info": FEATURE_HELP.get(feature_name, ""),
            }
        )

    comparison_df = pd.DataFrame(feature_rows)
    st.dataframe(
        comparison_df,
        hide_index=True,
        use_container_width=True,
    )


def _winner_text(flag: int) -> str:
    return "T" if int(flag) == 1 else "CT"


def _render_outcome_card(row: pd.Series) -> None:
    predicted = str(row["predicted_winner"])
    predicted_class = "t" if predicted == "T" else "ct"
    verdict = "Correct" if bool(row["model_correct"]) else "Miss"
    confidence = float(row["confidence_pct"])
    actual = str(row["actual_winner"])

    st.markdown(
        f"""
        <div class="outcome-card">
            <div class="subtle">Predicted winner</div>
            <div class="signal {predicted_class}">{predicted}</div>
            <div class="subtle">Actual winner: <strong>{actual}</strong> | Verdict: <strong>{verdict}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("T win probability", f"{confidence:.1f}%")
    st.progress(min(max(confidence / 100.0, 0.0), 1.0))


def _render_context(row: pd.Series) -> None:
    previous_winner = row["previous_round_winner"]
    previous_winner_text = "None" if int(previous_winner) == -1 else _winner_text(previous_winner)
    pistol_text = "Yes" if int(row["is_pistol_round"]) == 1 else "No"

    st.markdown(
        f"""
        <div class="hero-card">
            <h1>CS2 Round Outcome UI</h1>
            <div class="subtle">A round-start snapshot taken just after freeze end. Left: raw round data from the demo. Right: model output and the real result.</div>
            <div class="pill-row">
                <div class="pill"><strong>Match:</strong> {row['match_id']}</div>
                <div class="pill"><strong>Map:</strong> {row['map_name']}</div>
                <div class="pill"><strong>Round:</strong> {int(row['round_number'])}</div>
                <div class="pill"><strong>Pistol round:</strong> {pistol_text}</div>
                <div class="pill"><strong>Previous winner:</strong> {previous_winner_text}</div>
                <div class="pill"><strong>T streak:</strong> {int(row['t_win_streak'])}</div>
                <div class="pill"><strong>CT streak:</strong> {int(row['ct_win_streak'])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_round_explorer(dashboard_df: pd.DataFrame) -> None:
    mirage_df = dashboard_df[dashboard_df["map_name"].astype(str).str.lower() == "mirage"].copy()
    if mirage_df.empty:
        mirage_df = dashboard_df.copy()

    match_options = sorted(mirage_df["match_id"].dropna().unique().tolist())
    selected_match = st.selectbox("Match", match_options)

    filtered_match = mirage_df[mirage_df["match_id"] == selected_match].copy()
    filtered_match = filtered_match.sort_values("round_number").reset_index(drop=True)
    round_options = filtered_match["round_number"].astype(int).tolist()
    if not round_options:
        st.warning("No rounds found for the selected match.")
        return

    round_index_key = "selected_round_index"
    round_match_key = "selected_round_match"
    if st.session_state.get(round_match_key) != selected_match:
        st.session_state[round_match_key] = selected_match
        st.session_state[round_index_key] = 0

    current_round_index = int(st.session_state.get(round_index_key, 0))
    current_round_index = max(0, min(current_round_index, len(round_options) - 1))

    nav_col1, nav_col2, nav_col3 = st.columns([1.0, 1.2, 1.0], gap="medium")

    with nav_col1:
        previous_clicked = st.button(
            "Previous round",
            use_container_width=True,
            disabled=current_round_index <= 0,
        )
    if previous_clicked:
        current_round_index -= 1

    with nav_col2:
        st.markdown(
            f"""
            <div class="panel-card" style="padding: 0.85rem 1rem; text-align: center;">
                <div class="subtle">Current round</div>
                <div class="section-title" style="margin: 0.15rem 0 0 0;">{round_options[current_round_index]}</div>
                <div class="subtle">Round {current_round_index + 1} of {len(round_options)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col3:
        next_clicked = st.button(
            "Next round",
            use_container_width=True,
            disabled=current_round_index >= len(round_options) - 1,
        )
    if next_clicked:
        current_round_index += 1

    st.session_state[round_index_key] = current_round_index
    selected_round = round_options[current_round_index]

    selected_row = filtered_match[filtered_match["round_number"] == selected_round].iloc[0]

    _render_context(selected_row)

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.markdown("## Model Inputs")
        _render_core_feature_table(selected_row)

    with right_col:
        st.markdown("## Model Output")
        _render_outcome_card(selected_row)

        st.markdown("### Explanation Signals")
        st.markdown(
            f"""
            <div class="panel-card">
                <div><strong>Money diff:</strong> {int(selected_row['money_total_diff'])}</div>
                <div><strong>Equip diff:</strong> {int(selected_row['equipment_value_diff'])}</div>
                <div><strong>Utility diff:</strong> {int(selected_row['utility_total_diff'])}</div>
                <div><strong>Win streak diff:</strong> {int(selected_row['win_streak_diff'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Show raw round row"):
            st.dataframe(selected_row.to_frame().rename(columns={selected_row.name: "value"}))


def _add_train_test_split(dashboard_df: pd.DataFrame) -> pd.DataFrame:
    output = dashboard_df.copy()
    output["evaluation_split"] = "Train"

    if output["match_id"].astype(str).nunique() < 2:
        return output

    splitter = GroupShuffleSplit(
        n_splits=MAX_SPLIT_ATTEMPTS,
        test_size=GROUP_TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
    )
    X = output.loc[:, CORE_FEATURE_COLUMNS]
    y = output["won_round"].astype(int)
    groups = output["match_id"].astype(str)

    for train_idx, test_idx in splitter.split(X, y, groups):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        output.iloc[test_idx, output.columns.get_loc("evaluation_split")] = "Test"
        return output

    return output


def _safe_roc_auc(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.astype(int).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _safe_log_loss(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.astype(int).nunique() < 2:
        return None
    return float(log_loss(y_true, probabilities, labels=[0, 1]))


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _metric_row(label: str, frame: pd.DataFrame) -> dict[str, str | int]:
    y_true = frame["won_round"].astype(int)
    y_pred = frame["predicted_won_round"].astype(int)
    probabilities = frame["t_win_probability"].astype(float)
    return {
        "Split": label,
        "Rounds": len(frame),
        "Matches": frame["match_id"].astype(str).nunique(),
        "Accuracy": _format_metric(float(accuracy_score(y_true, y_pred))),
        "Precision": _format_metric(float(precision_score(y_true, y_pred, zero_division=0))),
        "Recall": _format_metric(float(recall_score(y_true, y_pred, zero_division=0))),
        "F1": _format_metric(float(f1_score(y_true, y_pred, zero_division=0))),
        "ROC-AUC": _format_metric(_safe_roc_auc(y_true, probabilities)),
        "Log loss": _format_metric(_safe_log_loss(y_true, probabilities)),
    }


def _render_confusion_matrix(frame: pd.DataFrame) -> None:
    matrix = confusion_matrix(
        frame["won_round"].astype(int),
        frame["predicted_won_round"].astype(int),
        labels=[0, 1],
    )
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual CT", "Actual T"],
        columns=["Predicted CT", "Predicted T"],
    )
    st.dataframe(matrix_df, use_container_width=True)


def _probability_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    binned = frame.copy()
    binned["probability_bin"] = pd.cut(
        binned["t_win_probability"],
        bins=[i / 10 for i in range(11)],
        include_lowest=True,
    ).astype(str)
    distribution = (
        binned.groupby(["probability_bin", "evaluation_split"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Train", "Test"], fill_value=0)
    )
    return distribution


def _render_evaluation(dashboard_df: pd.DataFrame, model_path: Path) -> None:
    evaluation_df = _add_train_test_split(dashboard_df)
    test_df = evaluation_df[evaluation_df["evaluation_split"] == "Test"].copy()
    train_df = evaluation_df[evaluation_df["evaluation_split"] == "Train"].copy()

    st.markdown(
        """
        <div class="hero-card">
            <h1>Model Evaluation</h1>
            <div class="subtle">Train/test results using the same match-aware split idea as training: complete matches are kept together, so test rounds come from held-out matches.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    checkpoint_metrics = load_neural_checkpoint(model_path).get("metrics", {})
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Saved test accuracy", _format_metric(checkpoint_metrics.get("accuracy")))
    metric_col2.metric("Saved test ROC-AUC", _format_metric(checkpoint_metrics.get("roc_auc")))
    metric_col3.metric("Saved test log loss", _format_metric(checkpoint_metrics.get("log_loss")))

    metric_rows = [_metric_row("Train", train_df)]
    if not test_df.empty:
        metric_rows.append(_metric_row("Test", test_df))
    metric_rows.append(_metric_row("All rounds", evaluation_df))

    st.markdown("## Metrics")
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)

    left_col, right_col = st.columns([1, 1], gap="large")
    with left_col:
        st.markdown("## Test Confusion Matrix")
        if test_df.empty:
            st.warning("No held-out test split could be reconstructed from the current dataset.")
        else:
            _render_confusion_matrix(test_df)

    with right_col:
        st.markdown("## T Win Probability Distribution")
        st.bar_chart(_probability_distribution(evaluation_df), use_container_width=True)

    st.markdown("## Accuracy Per Map")
    map_metrics = (
        evaluation_df.groupby(["map_name", "evaluation_split"], as_index=False)
        .agg(
            rounds=("won_round", "size"),
            accuracy=("model_correct", "mean"),
            t_win_rate=("won_round", "mean"),
        )
        .sort_values(["evaluation_split", "accuracy"], ascending=[True, False])
    )
    map_metrics["accuracy"] = map_metrics["accuracy"].round(3)
    map_metrics["t_win_rate"] = map_metrics["t_win_rate"].round(3)
    st.dataframe(map_metrics, hide_index=True, use_container_width=True)

    if not test_df.empty:
        st.markdown("## Held-Out Match Results")
        match_metrics = (
            test_df.groupby(["match_id", "map_name"], as_index=False)
            .agg(
                rounds=("won_round", "size"),
                accuracy=("model_correct", "mean"),
                average_t_probability=("t_win_probability", "mean"),
            )
            .sort_values("accuracy", ascending=True)
        )
        match_metrics["accuracy"] = match_metrics["accuracy"].round(3)
        match_metrics["average_t_probability"] = match_metrics["average_t_probability"].round(3)
        st.dataframe(match_metrics, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner=False)
def _load_experiment_log(log_path: str) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _render_experiments() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Experiment History</h1>
            <div class="subtle">Every training run can be logged here with its features, hyperparameters, dataset fingerprint, archived model path, and test metrics.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    experiments = _load_experiment_log(str(DEFAULT_EXPERIMENT_LOG_PATH))
    if experiments.empty:
        st.info(
            "No experiment runs logged yet. Train once with `python scripts/train_neural.py --run-name baseline --notes \"current features\"`."
        )
        return

    experiments = experiments.sort_values("created_at_utc", ascending=False).reset_index(drop=True)
    metric_columns = [
        "created_at_utc",
        "run_name",
        "notes",
        "dataset_rows",
        "dataset_matches",
        "feature_count",
        "accuracy",
        "roc_auc",
        "log_loss",
        "best_epoch",
        "hidden_sizes",
        "dropout",
        "learning_rate",
        "archived_model_path",
    ]
    visible_columns = [column for column in metric_columns if column in experiments.columns]
    display_df = experiments.loc[:, visible_columns].copy()
    for column in ["accuracy", "roc_auc", "log_loss", "dropout", "learning_rate"]:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").round(4)

    st.markdown("## Runs")
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.markdown("## Score Trend")
    trend_columns = [
        column for column in ["accuracy", "roc_auc", "log_loss"] if column in experiments.columns
    ]
    if trend_columns:
        trend_df = experiments.sort_values("created_at_utc").set_index("run_name")
        st.line_chart(trend_df[trend_columns], use_container_width=True)

    latest = experiments.iloc[0]
    st.markdown("## Latest Run Details")
    detail_rows = [
        {"Field": "Run ID", "Value": latest.get("run_id", "")},
        {"Field": "Notes", "Value": latest.get("notes", "")},
        {"Field": "Dataset fingerprint", "Value": latest.get("dataset_fingerprint", "")},
        {"Field": "Features", "Value": ", ".join(latest.get("feature_columns", []))},
        {"Field": "Archived model", "Value": latest.get("archived_model_path", "")},
    ]
    st.dataframe(pd.DataFrame(detail_rows), hide_index=True, use_container_width=True)


def main() -> None:
    _inject_styles()

    dataset_path = DEFAULT_DATASET_PATH
    model_path = DEFAULT_NEURAL_MODEL_PATH

    if not dataset_path.exists():
        st.error(
            "The full round dataset is missing. Run `python scripts/parse_all_demos.py` first."
        )
        return
    if not model_path.exists():
        st.error(
            "The neural model checkpoint is missing. Run `python scripts/train_neural.py` first."
        )
        return

    dashboard_df = _load_dashboard_frame(str(dataset_path), str(model_path))
    if dashboard_df.empty:
        st.warning("No rounds are available in the processed dataset yet.")
        return

    round_tab, evaluation_tab, experiments_tab = st.tabs(
        ["Round Explorer", "Evaluation", "Experiments"]
    )
    with round_tab:
        _render_round_explorer(dashboard_df)
    with evaluation_tab:
        _render_evaluation(dashboard_df, model_path)
    with experiments_tab:
        _render_experiments()


if __name__ == "__main__":
    main()
