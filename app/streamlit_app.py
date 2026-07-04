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
from src.generate_data import generate_candidate_actions, generate_material_pairs, save_demo_data
from src.gp_models import GPAbsoluteModel, GPDeltaModel
from src.local_linear import LocalLinearRegressor, PartialPoolingLocalLinearRegressor
from src.models import GlobalAbsoluteLinearModel, GlobalDeltaLinearModel
from src.plotting import plot_candidate_ranking, plot_local_contributions, plot_model_comparison, plot_uncertainty_vs_error


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


st.set_page_config(page_title="Baseline-Conditioned Model", layout="wide")
st.title("Baseline-Conditioned Material Change Demo")

pairs, candidates = load_data()
train, test, models, metrics, split_counts = train_models(pairs)

left, right = st.columns([0.36, 0.64])

with left:
    st.subheader("Base condition")
    base_id = st.selectbox("Base", pairs["base_id"].drop_duplicates().head(80))
    base_row = pairs[pairs["base_id"] == base_id].iloc[0]
    st.dataframe(base_row[["organization", "equipment", "product_family", "y_base", *CONTINUOUS_FEATURES]].to_frame("value"))

    st.subheader("Change")
    source = st.radio("Input mode", ["Manual deltas", "Candidate action"], horizontal=True)
    if source == "Candidate action":
        base_candidates = candidates[candidates["base_id"] == base_id]
        if base_candidates.empty:
            base_candidates = candidates.head(6)
        candidate_name = st.selectbox("Candidate", base_candidates["candidate_name"])
        query = base_candidates[base_candidates["candidate_name"] == candidate_name].head(1)
    else:
        deltas = {}
        for feature in CONTINUOUS_FEATURES:
            step = 5.0 if feature == "tempering_temp" else 0.01
            value = 0.0
            if feature == "C":
                value = 0.03
            if feature == "Mo":
                value = 0.08
            deltas[f"delta_{feature}"] = st.number_input(f"delta {feature}", value=value, step=step, format="%.4f")
        query = row_for_manual_delta(base_row, deltas)

    model_name = st.selectbox("Model", list(models))

with right:
    model = models[model_name]
    prediction = model.predict(query)
    st.metric("Predicted delta strength", f"{prediction.pred_delta_y[0]:+.1f} MPa")
    st.metric("Estimated uncertainty", f"{prediction.std_delta_y[0]:.1f} MPa")

    explanation = build_prediction_explanation(models["Local partial pooling"], query.iloc[0])
    st.write(explanation.text)
    if explanation.warnings:
        for warning in explanation.warnings:
            st.warning(warning)
    else:
        st.success("Support diagnostics look acceptable.")

    if hasattr(explanation.details, "contributions"):
        st.pyplot(plot_local_contributions(explanation.details.contributions))
        support_cols = st.columns(5)
        support_cols[0].metric("Effective sample size", f"{explanation.details.effective_sample_size:.1f}")
        support_cols[1].metric("Nearest examples", f"{len(explanation.details.nearest_examples)}")
        support_cols[2].metric("Same-org support", f"{explanation.details.same_organization_weight_ratio:.0%}")
        support_cols[3].metric("Avg base distance", f"{explanation.details.average_base_distance:.2f}")
        support_cols[4].metric("Direction support", f"{explanation.details.direction_coverage:.0%}")
        st.subheader("Nearest evidence")
        st.dataframe(explanation.details.nearest_examples)

    st.subheader("Model comparison")
    st.pyplot(plot_model_comparison(metrics))
    st.dataframe(pd.DataFrame(metrics).T.sort_values("rmse_delta_y"))
    st.caption(f"Split suite sizes: {split_counts}")

    st.subheader("Candidate comparison")
    scored = candidates[candidates["base_id"] == base_id].copy()
    if scored.empty:
        scored = candidates.head(8).copy()
    scored = score_candidates(models["Local partial pooling"], scored)
    st.caption(f"Ranking metrics: {candidate_ranking_metrics(models['Local partial pooling'], candidates)}")
    st.pyplot(plot_candidate_ranking(scored))
    uncertainty_frame = scored.rename(columns={"true_delta_y": "true_delta_y"})
    st.pyplot(plot_uncertainty_vs_error(uncertainty_frame))
    st.dataframe(
        scored[
            [
                "candidate_name",
                "pred_delta_y",
                "std_delta_y",
                "risk_adjusted_score",
                "same_org_support",
                "direction_support",
                "warning_count",
                "true_delta_y",
                "is_ood",
                "is_sparse_direction",
                *DELTA_FEATURES,
            ]
        ].sort_values("risk_adjusted_score", ascending=False)
    )
