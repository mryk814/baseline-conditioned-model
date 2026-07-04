import numpy as np

from src.evaluation import evaluate_models, split_random
from src.explanation import build_prediction_explanation
from src.generate_data import generate_candidate_actions, generate_material_pairs
from src.gp_models import GPAbsoluteModel, GPDeltaModel
from src.local_linear import LocalLinearRegressor, PartialPoolingLocalLinearRegressor
from src.models import GlobalAbsoluteLinearModel, GlobalDeltaLinearModel


def test_models_fit_and_predict_delta_with_uncertainty_shape():
    data = generate_material_pairs(n_baselines=30, actions_per_base=4, random_state=4)
    train, test = split_random(data, test_size=0.25, random_state=4)

    models = [
        GlobalAbsoluteLinearModel(),
        GlobalDeltaLinearModel(),
        LocalLinearRegressor(k_neighbors=35),
        PartialPoolingLocalLinearRegressor(k_neighbors=35),
        GPAbsoluteModel(max_train_size=80, random_state=4),
        GPDeltaModel(max_train_size=80, random_state=4),
    ]

    for model in models:
        model.fit(train)
        prediction = model.predict(test.head(6))
        assert len(prediction.pred_delta_y) == 6
        assert np.isfinite(prediction.pred_delta_y).all()
        assert len(prediction.std_delta_y) == 6
        assert (prediction.std_delta_y >= 0).all()


def test_local_linear_explanation_returns_support_metrics_and_contributions():
    data = generate_material_pairs(n_baselines=25, actions_per_base=4, random_state=5)
    query = data.iloc[[0]].copy()
    model = LocalLinearRegressor(k_neighbors=30).fit(data.iloc[5:].copy())

    explanation = model.explain_one(query.iloc[0])

    assert np.isfinite(explanation.pred_delta_y)
    assert explanation.effective_sample_size > 0
    assert 0 <= explanation.direction_coverage <= 1
    assert 0 <= explanation.same_organization_weight_ratio <= 1
    assert set(explanation.contributions) == set(model.delta_features)
    assert len(explanation.nearest_examples) <= 10


def test_evaluate_models_and_candidate_ranking_metrics_are_reported():
    data = generate_material_pairs(n_baselines=36, actions_per_base=4, random_state=8)
    candidates = generate_candidate_actions(data, candidates_per_base=5, random_state=8)
    train, test = split_random(data, test_size=0.25, random_state=8)
    model = GlobalDeltaLinearModel().fit(train)

    metrics = evaluate_models({"global_delta": model}, test)
    assert "global_delta" in metrics
    assert {"rmse_delta_y", "mae_delta_y", "r2_delta_y", "sign_accuracy"}.issubset(metrics["global_delta"])

    explanation = build_prediction_explanation(LocalLinearRegressor().fit(train), candidates.iloc[0])
    assert "model predicts" in explanation.text
    assert isinstance(explanation.warnings, list)
