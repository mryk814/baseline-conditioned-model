from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split


def split_random(data: pd.DataFrame, test_size: float = 0.25, random_state: int | None = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["product_family"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_ood(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = data[data["is_ood"] | data["is_sparse_direction"]].copy()
    train = data.drop(test.index).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _safe_r2(y_true: np.ndarray, pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, pred))


def split_baseline_ood(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_score = data.groupby("base_id")["is_ood"].max().rename("base_ood_score")
    enriched = data.join(base_score, on="base_id")
    test_ids = set(enriched[enriched["base_ood_score"]].base_id)
    if not test_ids:
        feature_cols = ["C", "Mn", "Cr", "Ni", "Mo", "tempering_temp", "holding_time", "cooling_rate"]
        centered = data[feature_cols] - data[feature_cols].mean()
        distances = np.sqrt(np.square(centered / data[feature_cols].std().replace(0, 1.0)).mean(axis=1))
        cutoff = distances.quantile(0.85)
        test_ids = set(data.loc[distances >= cutoff, "base_id"])
    test = data[data["base_id"].isin(test_ids)].copy()
    train = data[~data["base_id"].isin(test_ids)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_delta_direction_ood(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = data[data["is_sparse_direction"]].copy()
    if test.empty:
        test = data[data["is_ood"]].copy()
    train = data.drop(test.index).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_low_data_organization(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rare_org = data["organization"].value_counts().idxmin()
    org_rows = data[data["organization"] == rare_org]
    test_ids = set(org_rows["base_id"].drop_duplicates().sample(frac=0.5, random_state=42))
    test = data[data["base_id"].isin(test_ids)].copy()
    train = data[~data["base_id"].isin(test_ids)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_product_family_holdout(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout = data["product_family"].value_counts().idxmin()
    test = data[data["product_family"] == holdout].copy()
    train = data[data["product_family"] != holdout].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_suite(data: pd.DataFrame, random_state: int | None = 42) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {
        "random": split_random(data, random_state=random_state),
        "baseline_ood": split_baseline_ood(data),
        "delta_direction_ood": split_delta_direction_ood(data),
        "low_data_organization": split_low_data_organization(data),
        "product_family_holdout": split_product_family_holdout(data),
    }


def prediction_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    std: np.ndarray | None = None,
    y_new_true: np.ndarray | None = None,
    y_new_pred: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    metrics = {
        "rmse_delta_y": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae_delta_y": float(mean_absolute_error(y_true, pred)),
        "r2_delta_y": _safe_r2(y_true, pred),
        "sign_accuracy": float(np.mean(np.sign(y_true) == np.sign(pred))),
        "large_change_sign_accuracy": float(
            np.mean(np.sign(y_true[np.abs(y_true) >= np.quantile(np.abs(y_true), 0.75)]) == np.sign(pred[np.abs(y_true) >= np.quantile(np.abs(y_true), 0.75)]))
        ),
    }
    if y_new_true is not None and y_new_pred is not None:
        y_new_true = np.asarray(y_new_true, dtype=float)
        y_new_pred = np.asarray(y_new_pred, dtype=float)
        metrics["rmse_y_new"] = float(np.sqrt(mean_squared_error(y_new_true, y_new_pred)))
        metrics["mae_y_new"] = float(mean_absolute_error(y_new_true, y_new_pred))
        metrics["r2_y_new"] = _safe_r2(y_new_true, y_new_pred)
    if std is not None:
        std = np.maximum(np.asarray(std, dtype=float), 1e-9)
        metrics["interval_80_coverage"] = float(np.mean(np.abs(y_true - pred) <= 1.28 * std))
        metrics["interval_95_coverage"] = float(np.mean(np.abs(y_true - pred) <= 1.96 * std))
        metrics["mean_interval_80_width"] = float(np.mean(2.56 * std))
        metrics["mean_interval_95_width"] = float(np.mean(3.92 * std))
        metrics["negative_log_likelihood"] = float(np.mean(0.5 * np.log(2 * np.pi * np.square(std)) + np.square(y_true - pred) / (2 * np.square(std))))
    return metrics


def evaluate_models(models: dict[str, object], test: pd.DataFrame) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    y_true = test["delta_y"].to_numpy()
    for name, model in models.items():
        prediction = model.predict(test)
        metrics = prediction_metrics(
            y_true,
            prediction.pred_delta_y,
            prediction.std_delta_y,
            y_new_true=test["y_new"].to_numpy() if "y_new" in test else None,
            y_new_pred=prediction.pred_y_new,
        )
        explanations = prediction.details.get("explanations") if prediction.details else None
        if explanations:
            abs_error = np.abs(y_true - prediction.pred_delta_y)
            ess = np.array([item.effective_sample_size for item in explanations])
            direction = np.array([item.direction_coverage for item in explanations])
            base_distance = np.array([item.average_base_distance for item in explanations])
            same_org = np.array([item.same_organization_weight_ratio for item in explanations])
            metrics["error_vs_effective_sample_size_corr"] = _safe_corr(abs_error, ess)
            metrics["error_vs_direction_coverage_corr"] = _safe_corr(abs_error, direction)
            metrics["error_vs_nearest_neighbor_distance_corr"] = _safe_corr(abs_error, base_distance)
            metrics["error_vs_same_organization_weight_ratio_corr"] = _safe_corr(abs_error, same_org)
            if "is_ood" in test:
                risk_score = base_distance + (1.0 - direction)
                try:
                    metrics["ood_detection_auroc"] = float(roc_auc_score(test["is_ood"].astype(int), risk_score))
                except ValueError:
                    metrics["ood_detection_auroc"] = float("nan")
        else:
            metrics["error_vs_effective_sample_size_corr"] = float("nan")
            metrics["error_vs_direction_coverage_corr"] = float("nan")
            metrics["error_vs_nearest_neighbor_distance_corr"] = float("nan")
            metrics["error_vs_same_organization_weight_ratio_corr"] = float("nan")
            metrics["ood_detection_auroc"] = float("nan")
        results[name] = metrics
    return results


def evaluate_split_suite(
    data: pd.DataFrame,
    model_factories: dict[str, type | object],
    random_state: int | None = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    suite_results: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, (train, test) in split_suite(data, random_state=random_state).items():
        models = {}
        for model_name, factory in model_factories.items():
            model = factory() if isinstance(factory, type) else factory
            models[model_name] = model.fit(train)
        suite_results[split_name] = evaluate_models(models, test)
    return suite_results


def score_candidates(model: object, candidates: pd.DataFrame) -> pd.DataFrame:
    scored = candidates.copy()
    prediction = model.predict(scored)
    scored["pred_delta_y"] = prediction.pred_delta_y
    scored["std_delta_y"] = prediction.std_delta_y
    scored["risk_adjusted_score"] = scored["pred_delta_y"] - 0.75 * scored["std_delta_y"]
    explanations = prediction.details.get("explanations") if prediction.details else None
    if explanations:
        scored["same_org_support"] = [item.same_organization_weight_ratio for item in explanations]
        scored["direction_support"] = [item.direction_coverage for item in explanations]
        scored["warning_count"] = [len(item.warnings) for item in explanations]
    else:
        scored["same_org_support"] = np.nan
        scored["direction_support"] = np.nan
        scored["warning_count"] = np.nan
    scored["pred_rank"] = scored.groupby("base_id")["risk_adjusted_score"].rank(ascending=False, method="first")
    scored["true_rank"] = scored.groupby("base_id")["true_delta_y"].rank(ascending=False, method="first")
    return scored


def candidate_ranking_metrics(model: object, candidates: pd.DataFrame) -> dict[str, float]:
    scored = score_candidates(model, candidates)
    regrets = []
    top3_hits = []
    spearman_values = []
    score_quality_values = []
    for _, group in scored.groupby("base_id"):
        best_true = group["true_delta_y"].max()
        selected = group.sort_values("risk_adjusted_score", ascending=False).iloc[0]
        regrets.append(best_true - selected["true_delta_y"])
        top3_true_ids = set(group.sort_values("true_delta_y", ascending=False).head(3)["candidate_name"])
        top_pred = selected["candidate_name"]
        top3_hits.append(top_pred in top3_true_ids)
        spearman_values.append(group["pred_rank"].corr(group["true_rank"], method="spearman"))
        score_quality_values.append(group["risk_adjusted_score"].corr(group["true_delta_y"], method="spearman"))

    return {
        "top1_regret": float(np.mean(regrets)),
        "top3_hit_rate": float(np.mean(top3_hits)),
        "spearman_rank_correlation": float(np.nanmean(spearman_values)),
        "risk_adjusted_score_quality": float(np.nanmean(score_quality_values)),
    }
