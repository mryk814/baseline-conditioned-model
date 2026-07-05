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


st.set_page_config(page_title="ベース条件つき材料変更予測", layout="wide")

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
        ("組織", base_row["organization"]),
        ("設備", base_row["equipment"]),
        ("製品ファミリ", base_row["product_family"]),
        ("現在の強度", f"{base_row['y_base']:.1f} MPa"),
        ("C", f"{base_row['C']:.3f}"),
        ("Mn", f"{base_row['Mn']:.3f}"),
        ("Cr", f"{base_row['Cr']:.3f}"),
        ("Ni", f"{base_row['Ni']:.3f}"),
        ("Mo", f"{base_row['Mo']:.3f}"),
        ("焼戻し温度", f"{base_row['tempering_temp']:.1f}"),
        ("保持時間", f"{base_row['holding_time']:.2f}"),
        ("冷却速度", f"{base_row['cooling_rate']:.2f}"),
    ]
    return pd.DataFrame(rows, columns=["条件", "値"])


def strongest_contribution(contributions: dict[str, float]) -> str:
    if not contributions:
        return "n/a"
    feature, value = max(contributions.items(), key=lambda item: abs(item[1]))
    return display_feature(feature.replace("delta_", ""))


def risk_badge_class(risk: str) -> str:
    if risk == "根拠あり":
        return "badge-supported"
    if risk == "高リスク":
        return "badge-risk"
    return "badge-review"


def metric_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics).T
    columns = {
        "rmse_delta_y": "RMSE",
        "mae_delta_y": "MAE",
        "sign_accuracy": "符号正解率",
        "interval_95_coverage": "95%区間カバー率",
        "ood_detection_auroc": "OOD AUROC",
    }
    return frame[list(columns)].rename(columns=columns).sort_values("RMSE")


MODEL_LABELS = {
    "Global absolute linear": "全体・絶対値線形",
    "Global delta linear": "全体・変化量線形",
    "Local linear": "局所線形",
    "Local partial pooling": "局所線形 + 部分プーリング",
    "GP absolute": "GP・絶対値",
    "GP delta": "GP・変化量",
}

FEATURE_LABELS = {
    "C": "C",
    "Mn": "Mn",
    "Cr": "Cr",
    "Ni": "Ni",
    "Mo": "Mo",
    "tempering_temp": "焼戻し温度",
    "holding_time": "保持時間",
    "cooling_rate": "冷却速度",
}


def selected_model_key(label: str) -> str:
    inverse = {display: key for key, display in MODEL_LABELS.items()}
    return inverse[label]


def display_feature(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def evidence_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "pair_id": "事例ID",
            "organization": "組織",
            "product_family": "製品ファミリ",
            "observed_delta_y": "実測変化",
            "delta_C": "ΔC",
            "delta_Mn": "ΔMn",
            "delta_Cr": "ΔCr",
            "delta_Ni": "ΔNi",
            "delta_Mo": "ΔMo",
            "delta_tempering_temp": "Δ焼戻し温度",
            "delta_holding_time": "Δ保持時間",
            "delta_cooling_rate": "Δ冷却速度",
            "weight": "重み",
            "delta_similarity": "方向類似度",
            "base_distance": "ベース距離",
        }
    )


pairs, candidates = load_data()
train, test, models, metrics, split_counts = train_models(pairs)

st.title("材料変更の意思決定ワークベンチ")
st.markdown(
    "<div class='section-note'>ベース条件から少し条件を振ったとき、強度がどう変わりそうかと、その予測を支える根拠を確認します。</div>",
    unsafe_allow_html=True,
)

control_col, decision_col = st.columns([0.34, 0.66], gap="large")

with control_col:
    st.subheader("1. ベース条件を選ぶ")
    base_id = st.selectbox("ベース条件", pairs["base_id"].drop_duplicates().head(80))
    base_row = pairs[pairs["base_id"] == base_id].iloc[0]

    st.dataframe(base_summary_table(base_row), hide_index=True, width="stretch")

    st.subheader("2. 変更量を決める")
    source = st.radio("入力方法", ["手入力", "候補から選ぶ"], horizontal=True)
    if source == "候補から選ぶ":
        base_candidates = candidates[candidates["base_id"] == base_id]
        if base_candidates.empty:
            base_candidates = candidates.head(6)
        candidate_name = st.selectbox("候補", base_candidates["candidate_name"])
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
                deltas[f"delta_{feature}"] = st.number_input(
                    display_feature(feature),
                    value=value,
                    step=step,
                    format="%.4f",
                    key=f"delta_{feature}",
                )
        query = row_for_manual_delta(base_row, deltas)

    model_label = st.selectbox(
        "予測モデル",
        [MODEL_LABELS[key] for key in models],
        index=list(models).index("Local partial pooling"),
    )
    model_name = selected_model_key(model_label)

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
          <div class="decision-kicker">判断サマリー</div>
          <div class="decision-title">強度変化の予測: {prediction.pred_delta_y[0]:+.1f} MPa</div>
          <p class="decision-copy">{summary}</p>
          <div class="badge-row">
            <span class="badge {risk_badge_class(risk)}">{risk}</span>
            <span class="badge badge-review">信頼度: {confidence}</span>
            <span class="badge badge-supported">主な要因: {strongest_driver}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_items = [
        ("期待変化", f"{prediction.pred_delta_y[0]:+.1f} MPa"),
        ("不確実性", f"+/- {prediction.std_delta_y[0]:.1f}"),
        ("近傍根拠", f"{details.effective_sample_size:.1f} ESS"),
        ("同組織の根拠", f"{details.same_organization_weight_ratio:.0%}"),
        ("方向サポート", f"{details.direction_coverage:.0%}"),
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
        st.success("この変更は、近い過去事例に比較的よく支えられています。")

    st.pyplot(plot_local_contributions(details.contributions), width="stretch")

st.divider()

evidence_tab, candidates_tab, diagnostics_tab = st.tabs(["根拠", "候補ランキング", "モデル診断"])

with evidence_tab:
    st.subheader("予測を支える近傍事例")
    support_cols = st.columns(4)
    support_cols[0].metric("近傍事例数", f"{len(details.nearest_examples)}")
    support_cols[1].metric("ベース距離", f"{details.average_base_distance:.2f}")
    support_cols[2].metric("方向類似度", f"{details.average_delta_similarity:.2f}")
    support_cols[3].metric("推定不確実性", f"{details.std_delta_y:.1f}")
    st.dataframe(evidence_table(details.nearest_examples), width="stretch", hide_index=True)

with candidates_tab:
    st.subheader("候補変更の比較")
    scored = candidates[candidates["base_id"] == base_id].copy()
    if scored.empty:
        scored = candidates.head(8).copy()
    scored = score_candidates(models["Local partial pooling"], scored)
    ranking_metrics = candidate_ranking_metrics(models["Local partial pooling"], candidates)
    rank_cols = st.columns(4)
    rank_cols[0].metric("Top-1後悔", f"{ranking_metrics['top1_regret']:.1f}")
    rank_cols[1].metric("Top-3的中率", f"{ranking_metrics['top3_hit_rate']:.0%}")
    rank_cols[2].metric("順位相関", f"{ranking_metrics['spearman_rank_correlation']:.2f}")
    rank_cols[3].metric("スコア品質", f"{ranking_metrics['risk_adjusted_score_quality']:.2f}")
    st.pyplot(plot_candidate_ranking(scored), width="stretch")
    st.dataframe(format_candidate_table(scored.sort_values("risk_adjusted_score", ascending=False)), width="stretch", hide_index=True)

with diagnostics_tab:
    st.subheader("モデル診断")
    st.pyplot(plot_model_comparison(metrics), width="stretch")
    st.dataframe(metric_frame(metrics), width="stretch")
    scored_for_error = scored.rename(columns={"true_delta_y": "true_delta_y"})
    st.pyplot(plot_uncertainty_vs_error(scored_for_error), width="stretch")
    split_frame = pd.DataFrame(split_counts).T.reset_index().rename(columns={"index": "分割", "train": "学習件数", "test": "評価件数"})
    st.dataframe(split_frame, width="stretch", hide_index=True)
