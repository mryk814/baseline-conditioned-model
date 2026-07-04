from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import (
    candidate_ranking_metrics,
    evaluate_models,
    evaluate_split_suite,
    prediction_metrics,
    score_candidates,
    split_suite,
)
from src.generate_data import generate_candidate_actions, generate_material_pairs
from src.gp_models import GPAbsoluteModel
from src.local_linear import LocalLinearRegressor, PartialPoolingLocalLinearRegressor
from src.models import GlobalAbsoluteLinearModel, GlobalDeltaLinearModel
from src.plotting import (
    plot_candidate_ranking,
    plot_model_comparison,
    plot_predicted_vs_true,
    plot_uncertainty_vs_error,
)


def _small_models(train: pd.DataFrame) -> dict[str, object]:
    return {
        "absolute": GlobalAbsoluteLinearModel().fit(train),
        "delta": GlobalDeltaLinearModel().fit(train),
        "local": LocalLinearRegressor(k_neighbors=25).fit(train),
    }


def test_prediction_metrics_include_doc_required_accuracy_uncertainty_and_absolute_metrics():
    y_true = np.array([10.0, -12.0, 2.0, 35.0])
    pred = np.array([8.0, -8.0, -1.0, 30.0])
    std = np.array([3.0, 4.0, 2.0, 8.0])
    y_new_true = np.array([110.0, 90.0, 102.0, 135.0])
    y_new_pred = np.array([108.0, 94.0, 99.0, 130.0])

    metrics = prediction_metrics(y_true, pred, std, y_new_true=y_new_true, y_new_pred=y_new_pred)

    expected = {
        "rmse_delta_y",
        "mae_delta_y",
        "r2_delta_y",
        "rmse_y_new",
        "mae_y_new",
        "r2_y_new",
        "sign_accuracy",
        "large_change_sign_accuracy",
        "interval_80_coverage",
        "interval_95_coverage",
        "mean_interval_80_width",
        "mean_interval_95_width",
        "negative_log_likelihood",
    }
    assert expected.issubset(metrics)


def test_split_suite_covers_random_ood_low_data_and_product_family_holdout():
    data = generate_material_pairs(n_baselines=45, actions_per_base=4, random_state=31)

    splits = split_suite(data, random_state=31)

    assert {
        "random",
        "baseline_ood",
        "delta_direction_ood",
        "low_data_organization",
        "product_family_holdout",
    }.issubset(splits)
    for train, test in splits.values():
        assert len(train) > 0
        assert len(test) > 0


def test_evaluate_models_reports_applicability_diagnostics_and_split_suite():
    data = generate_material_pairs(n_baselines=36, actions_per_base=4, random_state=32)
    train, test = split_suite(data, random_state=32)["random"]
    models = _small_models(train)

    metrics = evaluate_models(models, test)
    assert {"error_vs_effective_sample_size_corr", "error_vs_direction_coverage_corr"}.issubset(metrics["local"])

    suite_metrics = evaluate_split_suite(data, {"delta": GlobalDeltaLinearModel, "local": LocalLinearRegressor}, random_state=32)
    assert {"random", "baseline_ood", "delta_direction_ood", "low_data_organization", "product_family_holdout"}.issubset(suite_metrics)
    assert "delta" in suite_metrics["random"]


def test_candidate_scoring_includes_ui_support_columns_and_rank_metrics():
    data = generate_material_pairs(n_baselines=30, actions_per_base=4, random_state=33)
    candidates = generate_candidate_actions(data, candidates_per_base=5, random_state=33)
    model = PartialPoolingLocalLinearRegressor(k_neighbors=25).fit(data)

    scored = score_candidates(model, candidates)
    assert {
        "same_org_support",
        "direction_support",
        "warning_count",
        "pred_rank",
        "true_rank",
        "risk_adjusted_score",
    }.issubset(scored.columns)

    metrics = candidate_ranking_metrics(model, candidates)
    assert {"top1_regret", "top3_hit_rate", "spearman_rank_correlation", "risk_adjusted_score_quality"}.issubset(metrics)


def test_local_explanation_exposes_evidence_distances_and_conflict_warnings():
    data = generate_material_pairs(n_baselines=30, actions_per_base=4, random_state=34)
    query = data.iloc[0].copy()
    for feature in ["delta_C", "delta_Mo", "delta_tempering_temp"]:
        query[feature] *= 6
    model = LocalLinearRegressor(k_neighbors=20).fit(data.iloc[5:])

    explanation = model.explain_one(query)

    assert explanation.average_base_distance >= 0
    assert -1 <= explanation.average_delta_similarity <= 1
    assert "base_distance" in explanation.nearest_examples.columns
    assert "observed_delta_y" in explanation.nearest_examples.columns
    assert any(flag in explanation.warnings for flag in ["EXTRAPOLATION_RISK", "CONFLICTING_LOCAL_EVIDENCE", "LOW_DIRECTION_SUPPORT"])


def test_gp_absolute_exposes_base_and_new_uncertainty_details():
    data = generate_material_pairs(n_baselines=18, actions_per_base=3, random_state=35)
    model = GPAbsoluteModel(max_train_size=60, random_state=35).fit(data)

    prediction = model.predict(data.head(4))

    assert {"pred_y_base", "std_y_base", "std_y_new"}.issubset(prediction.details)
    assert len(prediction.details["std_y_base"]) == 4
    assert len(prediction.details["std_y_new"]) == 4


def test_doc_required_plots_can_be_created():
    output_dir = Path("outputs/test-plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.png"):
        path.unlink()
    rows = pd.DataFrame(
        {
            "true_delta_y": [10, 20, -5, 8],
            "pred_delta_y": [12, 18, -2, 3],
            "std_delta_y": [2, 3, 4, 5],
            "candidate_name": ["a", "b", "c", "d"],
            "risk_adjusted_score": [10.5, 15.0, -5.0, -0.5],
        }
    )
    metrics = {"m1": {"rmse_delta_y": 2.0}, "m2": {"rmse_delta_y": 3.0}}

    figures = [
        plot_predicted_vs_true(rows, output_dir / "pred.png"),
        plot_model_comparison(metrics, output_dir / "models.png"),
        plot_uncertainty_vs_error(rows, output_dir / "uncertainty.png"),
        plot_candidate_ranking(rows, output_dir / "ranking.png"),
    ]

    assert all(fig is not None for fig in figures)
    assert len(list(output_dir.glob("*.png"))) == 4
