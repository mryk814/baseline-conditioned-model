from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import candidate_ranking_metrics, evaluate_models, score_candidates, split_random, split_suite
from src.explanation import build_prediction_explanation
from src.features import CONTINUOUS_FEATURES, DELTA_FEATURES
from src.generate_data import save_demo_data
from src.gp_models import GPAbsoluteModel, GPDeltaModel
from src.local_linear import LocalLinearRegressor, PartialPoolingLocalLinearRegressor
from src.models import GlobalAbsoluteLinearModel, GlobalDeltaLinearModel
from src.plotting import plot_candidate_ranking, plot_local_contributions, plot_model_comparison, plot_uncertainty_vs_error
from src.ui_text import (
    confidence_label,
    format_candidate_table,
    format_warning_message,
    risk_tone,
    top_decision_summary,
)


st.set_page_config(page_title="Baseline-Conditioned Model", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --surface: #111827;
      --surface-soft: #18212f;
      --surface-line: #2c3748;
      --ink: #f8fafc;
      --muted: #aeb8c6;
      --green: #5eb281;
      --amber: #d6a94a;
      --red: #d56b6b;
      --blue: #78a8d8;
    }
    .block-container {
      padding-top: 2.4rem;
      padding-bottom: 3rem;
      max-width: 1320px;
    }
    h1 {
      font-size: 2.15rem !important;
      letter-spacing: 0 !important;
      margin-bottom: 0.25rem !important;
    }
    h2, h3 {
      letter-spacing: 0 !important;
    }
    div[data-testid="stMetric"] {
      background: var(--surface-soft);
      border: 1px solid var(--surface-line);
      border-radius: 8px;
      padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {
      color: var(--muted);
    }
    .decision-band {
      background: linear-gradient(135deg, #121c2b 0%, #172337 100%);
      border: 1px solid var(--surface-line);
      border-radius: 8px;
      padding: 20px 22px;
      margin: 12px 0 18px 0;
    }
    .decision-kicker {
      color: var(--muted);
      font-size: 0.86rem;
      margin-bottom: 0.35rem;
    }
    .decision-title {
      color: var(--ink);
      font-size: 1.35rem;
      font-weight: 700;
      margin-bottom: 0.35rem;
    }
    .decision-copy {
      color: #d9e1ec;
      font-size: 0.98rem;
      line-height: 1.6;
      margin: 0;
    }
    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .badge {
      border-radius: 999px;
      display: inline-block;
      font-size: 0.82rem;
      font-weight: 650;
      padding: 5px 10px;
    }
    .badge-supported { background: rgba(94, 178, 129, 0.16); color: #9be0b8; border: 1px solid rgba(94, 178, 129, 0.36); }
    .badge-review { background: rgba(214, 169, 74, 0.16); color: #f0cc75; border: 1px solid rgba(214, 169, 74, 0.36); }
    .badge-risk { background: rgba(213, 107, 107, 0.16); color: #f0a3a3; border: 1px solid rgba(213, 107, 107, 0.36); }
    .section-note {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.55;
      margin-top: -0.3rem;
    }
    .warning-list {
      display: grid;
      gap: 8px;
      margin: 8px 0 4px 0;
    }
    .warning-item {
      border-left: 4px solid var(--amber);
      background: rgba(214, 169, 74, 0.12);
      border-radius: 6px;
      color: #f1dfb3;
      padding: 9px 12px;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin: 12px 0 18px 0;
    }
    .stat-card {
      background: var(--surface-soft);
      border: 1px solid var(--surface-line);
      border-radius: 8px;
      min-height: 86px;
      padding: 13px 15px;
    }
    .stat-label {
      color: var(--muted);
      font-size: 0.82rem;
      margin-bottom: 7px;
      white-space: normal;
    }
    .stat-value {
      color: var(--ink);
      font-size: 1.55rem;
      font-weight: 720;
      letter-spacing: 0;
      line-height: 1.15;
      overflow-wrap: anywhere;
    }
    @media (max-width: 1100px) {
      .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    .muted-small {
      color: var(--muted);
      font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs_path = ROOT / "data" / "materials_pairs.csv"
    candidates_path = ROOT / "data" / "candidate_actions.csv"
    if not pairs_path.exists() or not candidates_path.exists():
        save_demo_data(ROOT / "data")
    return pd.read_csv(pairs_path), pd.read_csv(candidates_path)


@st.cache_resource
def train_models(data: pd.DataFrame):
    train, test = split_random(data, random_state=11)
    models = {
        "Global absolute linear": GlobalAbsoluteLinearModel().fit(train),
        "Global delta linear": GlobalDeltaLinearModel().fit(train),
        "Local linear": LocalLinearRegressor().fit(train),
        "Local partial pooling": PartialPoolingLocalLinearRegressor().fit(train),
        "GP absolute": GPAbsoluteModel(max_train_size=260).fit(train),
        "GP delta": GPDeltaModel(max_train_size=260).fit(train),
    }
    split_counts = {name: {"train": len(parts[0]), "test": len(parts[1])} for name, parts in split_suite(data, random_state=11).items()}
    return train, test, models, evaluate_models(models, test), split_counts


def row_for_manual_delta(base_row: pd.Series, deltas: dict[str, float]) -> pd.DataFrame:
    row = base_row.copy()
    for feature in CONTINUOUS_FEATURES:
        row[f"delta_{feature}"] = deltas[f"delta_{feature}"]
        row[f"new_{feature}"] = row[feature] + deltas[f"delta_{feature}"]
    row["pair_id"] = f"{row['base_id']}-manual"
    row["delta_y"] = 0.0
    return pd.DataFrame([row])


def base_summary_table(base_row: pd.Series) -> pd.DataFrame:
    rows = [
        ("Organization", base_row["organization"]),
        ("Equipment", base_row["equipment"]),
        ("Product family", base_row["product_family"]),
        ("Current strength", f"{base_row['y_base']:.1f} MPa"),
        ("C", f"{base_row['C']:.3f}"),
        ("Mn", f"{base_row['Mn']:.3f}"),
        ("Cr", f"{base_row['Cr']:.3f}"),
        ("Ni", f"{base_row['Ni']:.3f}"),
        ("Mo", f"{base_row['Mo']:.3f}"),
        ("Tempering temp", f"{base_row['tempering_temp']:.1f}"),
        ("Holding time", f"{base_row['holding_time']:.2f}"),
        ("Cooling rate", f"{base_row['cooling_rate']:.2f}"),
    ]
    return pd.DataFrame(rows, columns=["Condition", "Value"])


def strongest_contribution(contributions: dict[str, float]) -> str:
    if not contributions:
        return "n/a"
    feature, value = max(contributions.items(), key=lambda item: abs(item[1]))
    return feature.replace("delta_", "")


def risk_badge_class(risk: str) -> str:
    if risk == "Supported":
        return "badge-supported"
    if risk == "High risk":
        return "badge-risk"
    return "badge-review"


def metric_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics).T
    columns = {
        "rmse_delta_y": "RMSE",
        "mae_delta_y": "MAE",
        "sign_accuracy": "Sign accuracy",
        "interval_95_coverage": "95% coverage",
        "ood_detection_auroc": "OOD AUROC",
    }
    return frame[list(columns)].rename(columns=columns).sort_values("RMSE")


pairs, candidates = load_data()
train, test, models, metrics, split_counts = train_models(pairs)

st.title("Material Change Decision Workbench")
st.markdown(
    "<div class='section-note'>Baseline condition, proposed perturbation, expected strength change, and the evidence behind the recommendation.</div>",
    unsafe_allow_html=True,
)

control_col, decision_col = st.columns([0.34, 0.66], gap="large")

with control_col:
    st.subheader("1. Define the decision")
    base_id = st.selectbox("Baseline", pairs["base_id"].drop_duplicates().head(80))
    base_row = pairs[pairs["base_id"] == base_id].iloc[0]

    st.dataframe(base_summary_table(base_row), hide_index=True, width="stretch")

    st.subheader("2. Set the change")
    source = st.radio("Change source", ["Manual deltas", "Candidate action"], horizontal=True)
    if source == "Candidate action":
        base_candidates = candidates[candidates["base_id"] == base_id]
        if base_candidates.empty:
            base_candidates = candidates.head(6)
        candidate_name = st.selectbox("Candidate", base_candidates["candidate_name"])
        query = base_candidates[base_candidates["candidate_name"] == candidate_name].head(1)
    else:
        deltas = {}
        composition_cols = st.columns(2)
        process_cols = st.columns(2)
        for index, feature in enumerate(CONTINUOUS_FEATURES):
            step = 5.0 if feature == "tempering_temp" else 0.01
            value = 0.0
            if feature == "C":
                value = 0.03
            if feature == "Mo":
                value = 0.08
            target_cols = process_cols if feature in {"tempering_temp", "holding_time", "cooling_rate"} else composition_cols
            with target_cols[index % 2]:
                deltas[f"delta_{feature}"] = st.number_input(feature, value=value, step=step, format="%.4f")
        query = row_for_manual_delta(base_row, deltas)

    model_name = st.selectbox("Prediction model", list(models), index=list(models).index("Local partial pooling"))

with decision_col:
    model = models[model_name]
    prediction = model.predict(query)
    explanation = build_prediction_explanation(models["Local partial pooling"], query.iloc[0])
    details = explanation.details
    strongest_driver = strongest_contribution(getattr(details, "contributions", {}))
    confidence = confidence_label(prediction.std_delta_y[0], getattr(details, "direction_coverage", 0.0))
    risk = risk_tone(getattr(details, "warnings", []))
    summary = top_decision_summary(
        pred_delta_y=prediction.pred_delta_y[0],
        std_delta_y=prediction.std_delta_y[0],
        confidence=confidence,
        risk=risk,
        strongest_driver=strongest_driver,
    )

    st.markdown(
        f"""
        <div class="decision-band">
          <div class="decision-kicker">Decision summary</div>
          <div class="decision-title">{prediction.pred_delta_y[0]:+.1f} MPa expected strength change</div>
          <p class="decision-copy">{summary}</p>
          <div class="badge-row">
            <span class="badge {risk_badge_class(risk)}">{risk}</span>
            <span class="badge badge-review">Confidence: {confidence}</span>
            <span class="badge badge-supported">Driver: {strongest_driver}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_items = [
        ("Expected change", f"{prediction.pred_delta_y[0]:+.1f} MPa"),
        ("Uncertainty", f"+/- {prediction.std_delta_y[0]:.1f}"),
        ("Evidence", f"{details.effective_sample_size:.1f} ESS"),
        ("Same-org", f"{details.same_organization_weight_ratio:.0%}"),
        ("Direction", f"{details.direction_coverage:.0%}"),
    ]
    st.markdown(
        "<div class='stat-grid'>"
        + "".join(
            f"<div class='stat-card'><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div></div>"
            for label, value in stat_items
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    warning_messages = [format_warning_message(code) for code in details.warnings]
    if warning_messages:
        st.markdown("<div class='warning-list'>" + "".join(f"<div class='warning-item'>{message}</div>" for message in warning_messages) + "</div>", unsafe_allow_html=True)
    else:
        st.success("Evidence support looks solid for this proposed change.")

    st.pyplot(plot_local_contributions(details.contributions), width="stretch")

st.divider()

evidence_tab, candidates_tab, diagnostics_tab = st.tabs(["Evidence", "Candidate ranking", "Model diagnostics"])

with evidence_tab:
    st.subheader("Evidence behind the estimate")
    support_cols = st.columns(4)
    support_cols[0].metric("Nearest examples", f"{len(details.nearest_examples)}")
    support_cols[1].metric("Average base distance", f"{details.average_base_distance:.2f}")
    support_cols[2].metric("Average direction similarity", f"{details.average_delta_similarity:.2f}")
    support_cols[3].metric("Estimated uncertainty", f"{details.std_delta_y:.1f}")
    st.dataframe(details.nearest_examples, width="stretch", hide_index=True)

with candidates_tab:
    st.subheader("Candidate ranking")
    scored = candidates[candidates["base_id"] == base_id].copy()
    if scored.empty:
        scored = candidates.head(8).copy()
    scored = score_candidates(models["Local partial pooling"], scored)
    ranking_metrics = candidate_ranking_metrics(models["Local partial pooling"], candidates)
    rank_cols = st.columns(4)
    rank_cols[0].metric("Top-1 regret", f"{ranking_metrics['top1_regret']:.1f}")
    rank_cols[1].metric("Top-3 hit rate", f"{ranking_metrics['top3_hit_rate']:.0%}")
    rank_cols[2].metric("Rank correlation", f"{ranking_metrics['spearman_rank_correlation']:.2f}")
    rank_cols[3].metric("Score quality", f"{ranking_metrics['risk_adjusted_score_quality']:.2f}")
    st.pyplot(plot_candidate_ranking(scored), width="stretch")
    st.dataframe(format_candidate_table(scored.sort_values("risk_adjusted_score", ascending=False)), width="stretch", hide_index=True)

with diagnostics_tab:
    st.subheader("Model diagnostics")
    st.pyplot(plot_model_comparison(metrics), width="stretch")
    st.dataframe(metric_frame(metrics), width="stretch")
    scored_for_error = scored.rename(columns={"true_delta_y": "true_delta_y"})
    st.pyplot(plot_uncertainty_vs_error(scored_for_error), width="stretch")
    split_frame = pd.DataFrame(split_counts).T.reset_index().rename(columns={"index": "Split"})
    st.dataframe(split_frame, width="stretch", hide_index=True)
