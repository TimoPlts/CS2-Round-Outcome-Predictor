from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import altair as alt
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
            --bg: #dfe8ef;
            --paper: #ffffff;
            --ink: #0b1f33;
            --muted: #40566b;
            --navy: #183b56;
            --tint-t: #c45137;
            --tint-ct: #315f86;
            --accent: #14705c;
            --border: #b8c8d4;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(196, 81, 55, 0.12), transparent 32%),
                radial-gradient(circle at top right, rgba(49, 95, 134, 0.14), transparent 28%),
                linear-gradient(180deg, #edf2f6 0%, var(--bg) 100%);
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            font-family: "Trebuchet MS", "Gill Sans", sans-serif;
            letter-spacing: 0.02em;
            color: var(--ink);
        }
        p, label, [data-testid="stMarkdownContainer"] {
            color: var(--ink);
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.35rem;
            padding: 0.35rem;
            border-radius: 14px;
            background: var(--navy);
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] {
            height: 2.8rem;
            padding: 0 1rem;
            border-radius: 10px;
            font-weight: 700;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"],
        [data-testid="stTabs"] button[data-baseweb="tab"] p {
            color: #ffffff !important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"]:hover,
        [data-testid="stTabs"] button[data-baseweb="tab"]:hover p {
            background: rgba(255, 255, 255, 0.10);
            color: #ffffff !important;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {
            background: #ffffff;
            color: var(--navy) !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--tint-t);
        }
        .hero-card, .panel-card, .outcome-card {
            background: var(--paper);
            border: 1px solid var(--border);
            border-radius: 22px;
            box-shadow: 0 16px 36px rgba(36, 59, 83, 0.10);
        }
        .hero-card {
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.2rem;
        }
        .summary-card {
            padding: 1.1rem 1.25rem;
            margin: 0.3rem 0 1.2rem 0;
            border-left: 6px solid var(--accent);
            border-radius: 14px;
            background: #f7fbfa;
            box-shadow: 0 8px 20px rgba(24, 59, 86, 0.08);
        }
        .summary-card strong {
            color: var(--navy);
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
            background: #edf3f7;
            border: 1px solid var(--border);
            font-size: 0.85rem;
        }
        [data-testid="stMetric"] {
            min-height: 7.5rem;
            padding: 1rem 1.15rem;
            border: 1px solid var(--border);
            border-left: 5px solid var(--accent);
            border-radius: 16px;
            background: var(--paper);
            box-shadow: 0 10px 24px rgba(36, 59, 83, 0.08);
        }
        [data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 700;
        }
        [data-testid="stMetricValue"] {
            color: var(--navy);
            font-weight: 800;
        }
        [data-testid="stDataFrame"],
        [data-testid="stVegaLiteChart"],
        [data-testid="stArrowVegaLiteChart"] {
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--paper);
            box-shadow: 0 8px 20px rgba(24, 59, 86, 0.12);
        }
        [data-testid="stDataFrame"] {
            outline: 3px solid rgba(255, 255, 255, 0.48);
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
            box-shadow: 0 10px 24px rgba(36, 59, 83, 0.08);
        }
        div.stButton > button:hover {
            border-color: var(--accent);
            color: var(--ink);
            background: #f5faf8;
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
        [data-testid="stSelectbox"] label,
        [data-testid="stMultiSelect"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stExpander"] summary,
        [data-testid="stAlert"] p {
            color: var(--ink);
        }
        [data-testid="stExpander"] {
            border-color: var(--border);
            background: rgba(255, 255, 255, 0.72);
        }
        @media (max-width: 760px) {
            .block-container {
                padding: 1rem 0.75rem 2rem 0.75rem;
            }
            [data-testid="stTabs"] [data-baseweb="tab-list"] {
                overflow-x: auto;
            }
            [data-testid="stTabs"] button[data-baseweb="tab"] {
                min-width: max-content;
                padding: 0 0.75rem;
            }
            .hero-card, .panel-card, .outcome-card {
                border-radius: 14px;
            }
            h1 {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_dashboard_frame(
    dataset_path: str,
    model_path: str,
    dataset_modified_ns: int,
    model_modified_ns: int,
) -> pd.DataFrame:
    del dataset_modified_ns, model_modified_ns
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
        value = f"{float(raw_value):g}"
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
        width="stretch",
    )


def _winner_text(flag: int) -> str:
    return "T" if int(flag) == 1 else "CT"


def _render_outcome_card(row: pd.Series) -> None:
    predicted = str(row["predicted_winner"])
    predicted_class = "t" if predicted == "T" else "ct"
    verdict = "Correct" if bool(row["model_correct"]) else "Miss"
    confidence = float(row["confidence_pct"])
    winner_confidence = max(confidence, 100.0 - confidence)
    confidence_label = "uncertain" if 40.0 <= confidence <= 60.0 else "confident"
    actual = str(row["actual_winner"])

    st.markdown(
        f"""
        <div class="outcome-card">
            <div class="subtle">Predicted winner</div>
            <div class="signal {predicted_class}">{predicted}</div>
            <div class="subtle">Actual winner: <strong>{actual}</strong> | Verdict: <strong>{verdict}</strong></div>
            <div class="subtle">Prediction confidence: <strong>{winner_confidence:.1f}%</strong> ({confidence_label})</div>
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


def _change_round_index(index_key: str, step: int, last_index: int) -> None:
    current_index = int(st.session_state.get(index_key, 0))
    st.session_state[index_key] = max(0, min(current_index + step, last_index))


def _render_round_explorer(dashboard_df: pd.DataFrame) -> None:
    map_options = sorted(dashboard_df["map_name"].dropna().astype(str).unique().tolist())
    selected_map = st.selectbox("Map", map_options)
    map_df = dashboard_df[dashboard_df["map_name"].astype(str) == selected_map].copy()

    match_options = sorted(map_df["match_id"].dropna().unique().tolist())
    selected_match = st.selectbox("Match", match_options)

    filtered_match = map_df[map_df["match_id"] == selected_match].copy()
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
    st.session_state[round_index_key] = current_round_index
    last_round_index = len(round_options) - 1

    nav_col1, nav_col2, nav_col3 = st.columns([1.0, 1.2, 1.0], gap="medium")

    with nav_col1:
        st.button(
            "Previous round",
            width="stretch",
            disabled=current_round_index <= 0,
            on_click=_change_round_index,
            args=(round_index_key, -1, last_round_index),
        )

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
        st.button(
            "Next round",
            width="stretch",
            disabled=current_round_index >= last_round_index,
            on_click=_change_round_index,
            args=(round_index_key, 1, last_round_index),
        )

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
            raw_row = selected_row.astype(str).to_frame(name="value")
            st.dataframe(raw_row, width="stretch")


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


def _metric_values(frame: pd.DataFrame) -> dict[str, float | None]:
    y_true = frame["won_round"].astype(int)
    y_pred = frame["predicted_won_round"].astype(int)
    probabilities = frame["t_win_probability"].astype(float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, probabilities),
        "log_loss": _safe_log_loss(y_true, probabilities),
    }


def _render_model_summary(frame: pd.DataFrame) -> None:
    metrics = _metric_values(frame)
    matrix = confusion_matrix(
        frame["won_round"].astype(int),
        frame["predicted_won_round"].astype(int),
        labels=[0, 1],
    )
    ct_recall = matrix[0, 0] / matrix[0].sum() if matrix[0].sum() else 0.0
    t_recall = matrix[1, 1] / matrix[1].sum() if matrix[1].sum() else 0.0
    stronger_side = "CT" if ct_recall >= t_recall else "T"
    stronger_recall = max(ct_recall, t_recall)
    weaker_side = "T" if stronger_side == "CT" else "CT"
    uncertain_rate = frame["t_win_probability"].between(0.4, 0.6).mean()
    roc_auc_text = _format_metric(metrics["roc_auc"])

    st.markdown(
        f"""
        <div class="summary-card">
            <strong>Model summary.</strong>
            On these {len(frame):,} rounds, the model predicts
            <strong>{metrics['accuracy']:.1%}</strong> correctly with a ROC-AUC of
            <strong>{roc_auc_text}</strong>. It recognizes {stronger_side} wins
            best ({stronger_recall:.1%} recall), while {weaker_side} wins remain the larger
            opportunity. The model is uncertain on <strong>{uncertain_rate:.1%}</strong>
            of rounds.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_confusion_matrix(frame: pd.DataFrame) -> None:
    matrix = confusion_matrix(
        frame["won_round"].astype(int),
        frame["predicted_won_round"].astype(int),
        labels=[0, 1],
    )
    rows = []
    for actual_index, actual in enumerate(["CT", "T"]):
        row_total = matrix[actual_index].sum()
        for predicted_index, predicted in enumerate(["CT", "T"]):
            count = int(matrix[actual_index, predicted_index])
            rows.append(
                {
                    "Actual": actual,
                    "Predicted": predicted,
                    "Rounds": count,
                    "Percent": count / row_total if row_total else 0.0,
                    "Label": f"{count}\n({count / row_total:.1%})" if row_total else str(count),
                }
            )
    matrix_df = pd.DataFrame(rows)
    heatmap = (
        alt.Chart(matrix_df)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X("Predicted:N", title="Predicted winner", sort=["CT", "T"]),
            y=alt.Y("Actual:N", title="Actual winner", sort=["CT", "T"]),
            color=alt.Color(
                "Percent:Q",
                scale=alt.Scale(domain=[0, 1], range=["#edf4f8", "#075f9f"]),
                legend=alt.Legend(format=".0%", title="Row share"),
            ),
            tooltip=[
                "Actual:N",
                "Predicted:N",
                "Rounds:Q",
                alt.Tooltip("Percent:Q", format=".1%"),
            ],
        )
    )
    labels = (
        alt.Chart(matrix_df)
        .mark_text(fontSize=16, fontWeight="bold")
        .encode(
            x=alt.X("Predicted:N", sort=["CT", "T"]),
            y=alt.Y("Actual:N", sort=["CT", "T"]),
            text="Label:N",
            color=alt.condition(alt.datum.Percent > 0.5, alt.value("white"), alt.value("#0b1f33")),
        )
    )
    st.altair_chart(_style_chart((heatmap + labels).properties(height=300)), width="stretch")


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


def _style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            domainColor="#405f76",
            domainWidth=1.2,
            grid=True,
            gridColor="#a9bdcc",
            gridOpacity=0.9,
            gridWidth=1.1,
            labelColor="#29465b",
            tickColor="#405f76",
            titleColor="#0b1f33",
        )
        .configure_legend(
            labelColor="#29465b",
            titleColor="#0b1f33",
        )
        .configure_view(
            fill="#ffffff",
            stroke="#9fb5c5",
            strokeWidth=1,
        )
    )


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
    metric_col1.metric(
        "Saved test accuracy",
        _format_metric(checkpoint_metrics.get("accuracy")),
        help="Share of held-out rounds where the predicted winner was correct.",
    )
    metric_col2.metric(
        "Saved test ROC-AUC",
        _format_metric(checkpoint_metrics.get("roc_auc")),
        help="How well the model ranks T wins above CT wins. 0.5 is random; 1.0 is perfect.",
    )
    metric_col3.metric(
        "Saved test log loss",
        _format_metric(checkpoint_metrics.get("log_loss")),
        help="Penalizes confident wrong predictions. Lower is better.",
    )

    summary_df = test_df if not test_df.empty else evaluation_df
    _render_model_summary(summary_df)

    st.markdown("## Explore Results")
    filter_col1, filter_col2 = st.columns([2, 1])
    map_options = sorted(evaluation_df["map_name"].dropna().astype(str).unique().tolist())
    selected_maps = filter_col1.multiselect(
        "Maps",
        map_options,
        default=map_options,
        help="All evaluation sections below update with this selection.",
    )
    selected_split = filter_col2.selectbox("Split", ["All rounds", "Test", "Train"])

    filtered_df = evaluation_df[evaluation_df["map_name"].astype(str).isin(selected_maps)].copy()
    if selected_split != "All rounds":
        filtered_df = filtered_df[filtered_df["evaluation_split"] == selected_split].copy()
    if filtered_df.empty:
        st.warning("No rounds match the selected filters.")
        return

    filtered_metrics = _metric_values(filtered_df)
    uncertain_rounds = int(filtered_df["t_win_probability"].between(0.4, 0.6).sum())
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    overview_col1.metric(
        "Filtered accuracy",
        _format_metric(filtered_metrics["accuracy"]),
        help="Correct predictions divided by all selected rounds.",
    )
    overview_col2.metric(
        "Filtered ROC-AUC",
        _format_metric(filtered_metrics["roc_auc"]),
        help="Ranking quality for the selected rounds. 0.5 is random; 1.0 is perfect.",
    )
    overview_col3.metric(
        "Filtered log loss",
        _format_metric(filtered_metrics["log_loss"]),
        help="Confidence-sensitive error for the selected rounds. Lower is better.",
    )
    overview_col4.metric(
        "Uncertain rounds",
        f"{uncertain_rounds} ({uncertain_rounds / len(filtered_df):.1%})",
        help="Rounds where the predicted T win probability is between 40% and 60%.",
    )

    metric_rows = []
    for split_name in ["Train", "Test"]:
        split_frame = filtered_df[filtered_df["evaluation_split"] == split_name]
        if not split_frame.empty:
            metric_rows.append(_metric_row(split_name, split_frame))
    metric_rows.append(_metric_row("Selected rounds", filtered_df))

    st.markdown("## Metrics")
    st.dataframe(
        pd.DataFrame(metric_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Accuracy": st.column_config.TextColumn(help="Share of correct predictions."),
            "Precision": st.column_config.TextColumn(help="When T is predicted, how often T wins."),
            "Recall": st.column_config.TextColumn(help="Share of actual T wins found by the model."),
            "F1": st.column_config.TextColumn(help="Balance between T precision and recall."),
            "ROC-AUC": st.column_config.TextColumn(help="Ranking quality; higher is better."),
            "Log loss": st.column_config.TextColumn(help="Confidence-sensitive error; lower is better."),
        },
    )
    with st.expander("How to read these metrics"):
        st.markdown(
            """
            - **Accuracy** is the overall share of correct round-winner predictions.
            - **Precision** answers: when the model predicts T, how often does T actually win?
            - **Recall** answers: of all actual T wins, how many did the model find?
            - **F1** balances precision and recall.
            - **ROC-AUC** measures ranking quality across every possible classification threshold.
            - **Log loss** rewards calibrated probabilities and strongly penalizes confident mistakes.
            """
        )

    left_col, right_col = st.columns([1, 1], gap="large")
    with left_col:
        st.markdown("## Confusion Matrix")
        _render_confusion_matrix(filtered_df)

    with right_col:
        st.markdown("## T Win Probability Distribution")
        distribution = (
            _probability_distribution(filtered_df)
            .reset_index()
            .melt(
                id_vars="probability_bin",
                var_name="Split",
                value_name="Rounds",
            )
        )
        probability_chart = (
            alt.Chart(distribution)
            .mark_bar()
            .encode(
                x=alt.X("probability_bin:N", title="T win probability"),
                y=alt.Y("Rounds:Q", title="Rounds"),
                color=alt.Color(
                    "Split:N",
                    scale=alt.Scale(
                        domain=["Train", "Test"],
                        range=["#55a9df", "#075f9f"],
                    ),
                ),
                tooltip=["probability_bin:N", "Split:N", "Rounds:Q"],
            )
            .properties(height=330)
        )
        st.altair_chart(_style_chart(probability_chart), width="stretch")

    st.markdown("## Accuracy Per Map")
    map_metrics = (
        filtered_df.groupby(["map_name", "evaluation_split"], as_index=False)
        .agg(
            rounds=("won_round", "size"),
            accuracy=("model_correct", "mean"),
            t_win_rate=("won_round", "mean"),
        )
        .sort_values(["evaluation_split", "accuracy"], ascending=[True, False])
    )
    map_metrics["accuracy"] = map_metrics["accuracy"].round(3)
    map_metrics["t_win_rate"] = map_metrics["t_win_rate"].round(3)
    st.dataframe(map_metrics, hide_index=True, width="stretch")

    filtered_test_df = filtered_df[filtered_df["evaluation_split"] == "Test"]
    if not filtered_test_df.empty:
        st.markdown("## Held-Out Match Results")
        match_metrics = (
            filtered_test_df.groupby(["match_id", "map_name"], as_index=False)
            .agg(
                rounds=("won_round", "size"),
                accuracy=("model_correct", "mean"),
                average_t_probability=("t_win_probability", "mean"),
            )
            .sort_values("accuracy", ascending=True)
        )
        match_metrics["accuracy"] = match_metrics["accuracy"].round(3)
        match_metrics["average_t_probability"] = match_metrics["average_t_probability"].round(3)
        st.dataframe(match_metrics, hide_index=True, width="stretch")


@st.cache_data(show_spinner=False)
def _load_experiment_log(log_path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
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


def _display_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value)


def _render_experiment_comparison(experiments: pd.DataFrame) -> None:
    if len(experiments) < 2:
        st.info("Log at least two experiment runs to enable comparison.")
        return

    labels = [
        f"{row['run_name']} | {str(row['created_at_utc'])[:19]}"
        for _, row in experiments.iterrows()
    ]
    compare_col1, compare_col2 = st.columns(2)
    candidate_label = compare_col1.selectbox("Candidate run", labels, index=0)
    baseline_label = compare_col2.selectbox("Baseline run", labels, index=1)
    candidate = experiments.iloc[labels.index(candidate_label)]
    baseline = experiments.iloc[labels.index(baseline_label)]

    score_columns = [
        column for column in ["accuracy", "roc_auc", "log_loss"] if column in experiments.columns
    ]
    score_cards = st.columns(len(score_columns))
    for card, column in zip(score_cards, score_columns):
        candidate_score = float(candidate[column])
        baseline_score = float(baseline[column])
        card.metric(
            column.replace("_", " ").title(),
            f"{candidate_score:.4f}",
            delta=f"{candidate_score - baseline_score:+.4f} vs baseline",
            delta_color="inverse" if column == "log_loss" else "normal",
        )

    comparison_fields = [
        "notes",
        "dataset_rows",
        "dataset_matches",
        "feature_count",
        "hidden_sizes",
        "dropout",
        "learning_rate",
        "batch_size",
        "best_epoch",
        "epochs_completed",
        "device",
    ]
    comparison_rows = [
        {
            "Setting": field.replace("_", " ").title(),
            "Candidate": _display_value(candidate.get(field, "")),
            "Baseline": _display_value(baseline.get(field, "")),
        }
        for field in comparison_fields
        if field in experiments.columns
    ]
    st.dataframe(
        pd.DataFrame(comparison_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Setting": st.column_config.TextColumn(width="small"),
            "Candidate": st.column_config.TextColumn(width="large"),
            "Baseline": st.column_config.TextColumn(width="large"),
        },
    )

    candidate_features = set(candidate.get("feature_columns", []) or [])
    baseline_features = set(baseline.get("feature_columns", []) or [])
    added_features = sorted(candidate_features - baseline_features)
    removed_features = sorted(baseline_features - candidate_features)
    if added_features or removed_features:
        st.markdown(
            f"**Feature changes:** added `{', '.join(added_features) or 'none'}`; "
            f"removed `{', '.join(removed_features) or 'none'}`."
        )
    else:
        st.caption("Both runs use the same feature set.")


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

    refresh_col, _ = st.columns([1, 5])
    with refresh_col:
        if st.button("Refresh runs", use_container_width=True):
            _load_experiment_log.clear()

    log_modified_ns = (
        DEFAULT_EXPERIMENT_LOG_PATH.stat().st_mtime_ns
        if DEFAULT_EXPERIMENT_LOG_PATH.exists()
        else 0
    )
    experiments = _load_experiment_log(
        str(DEFAULT_EXPERIMENT_LOG_PATH),
        log_modified_ns,
    )
    if experiments.empty:
        st.info(
            "No experiment runs logged yet. Train once with `python scripts/train_neural.py --run-name baseline --notes \"current features\"`."
        )
        return

    experiments = experiments.sort_values("created_at_utc", ascending=False).reset_index(drop=True)
    st.markdown("## Compare Runs")
    _render_experiment_comparison(experiments)

    st.markdown("## Runs")
    search_query = st.text_input(
        "Search runs",
        placeholder="Search by run name or notes",
    ).strip()
    visible_experiments = experiments
    if search_query:
        searchable = (
            experiments["run_name"].fillna("").astype(str)
            + " "
            + experiments["notes"].fillna("").astype(str)
        )
        visible_experiments = experiments[
            searchable.str.contains(search_query, case=False, regex=False)
        ].copy()
    if visible_experiments.empty:
        st.info("No experiment runs match this search.")

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
    display_df = visible_experiments.loc[:, visible_columns].copy()
    for column in ["accuracy", "roc_auc", "log_loss", "dropout", "learning_rate"]:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").round(4)

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        column_config={
            "run_name": st.column_config.TextColumn("Run", width="medium"),
            "notes": st.column_config.TextColumn("Notes", width="large"),
            "created_at_utc": st.column_config.TextColumn("Created", width="medium"),
            "archived_model_path": st.column_config.TextColumn("Archived model", width="large"),
        },
    )

    st.markdown("## Score Trend")
    trend_columns = [
        column for column in ["accuracy", "roc_auc", "log_loss"] if column in experiments.columns
    ]
    if trend_columns and not visible_experiments.empty:
        trend_df = (
            visible_experiments.sort_values("created_at_utc")
            .loc[:, ["run_name", "created_at_utc", "notes", *trend_columns]]
            .melt(
                id_vars=["run_name", "created_at_utc", "notes"],
                var_name="Metric",
                value_name="Score",
            )
        )
        trend_chart = (
            alt.Chart(trend_df)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=65), strokeWidth=3)
            .encode(
                x=alt.X(
                    "run_name:N",
                    title="Run",
                    axis=alt.Axis(labelAngle=-45),
                ),
                y=alt.Y(
                    "Score:Q",
                    title="Score",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "Metric:N",
                    scale=alt.Scale(
                        domain=["accuracy", "log_loss", "roc_auc"],
                        range=["#075f9f", "#55a9df", "#c45137"],
                    ),
                ),
                tooltip=[
                    "run_name:N",
                    "created_at_utc:N",
                    "notes:N",
                    "Metric:N",
                    alt.Tooltip("Score:Q", format=".4f"),
                ],
            )
            .properties(height=380)
        )
        st.altair_chart(_style_chart(trend_chart), width="stretch")

    latest = experiments.iloc[0]
    st.markdown("## Latest Run Details")
    detail_fields = [
        ("Run ID", "run_id"),
        ("Notes", "notes"),
        ("Dataset fingerprint", "dataset_fingerprint"),
        ("Features", "feature_columns"),
        ("Archived model", "archived_model_path"),
    ]
    for label, field in detail_fields:
        with st.expander(label, expanded=label == "Notes"):
            st.write(_display_value(latest.get(field, "")) or "Not recorded")


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

    dashboard_df = _load_dashboard_frame(
        str(dataset_path),
        str(model_path),
        dataset_path.stat().st_mtime_ns,
        model_path.stat().st_mtime_ns,
    )
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
