from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def split_random(data: pd.DataFrame, test_size: float = 0.25, random_state: int | None = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data["product_family"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_ood(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = data[data["is_ood"] | data["is_sparse_direction"]].copy()
    train = data.drop(test.index).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prediction_metrics(y_true: np.ndarray, pred: np.ndarray, std: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "rmse_delta_y": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae_delta_y": float(mean_absolute_error(y_true, pred)),
        "r2_delta_y": float(r2_score(y_true, pred)),
        "sign_accuracy": float(np.mean(np.sign(y_true) == np.sign(pred))),
    }
    if std is not None:
        metrics["interval_80_coverage"] = float(np.mean(np.abs(y_true - pred) <= 1.28 * np.maximum(std, 1e-9)))
        metrics["mean_interval_80_width"] = float(np.mean(2.56 * np.maximum(std, 1e-9)))
    return metrics


def evaluate_models(models: dict[str, object], test: pd.DataFrame) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    y_true = test["delta_y"].to_numpy()
    for name, model in models.items():
        prediction = model.predict(test)
        results[name] = prediction_metrics(y_true, prediction.pred_delta_y, prediction.std_delta_y)
    return results


def candidate_ranking_metrics(model: object, candidates: pd.DataFrame) -> dict[str, float]:
    scored = candidates.copy()
    prediction = model.predict(scored)
    scored["pred_delta_y"] = prediction.pred_delta_y
    scored["std_delta_y"] = prediction.std_delta_y
    scored["risk_adjusted_score"] = scored["pred_delta_y"] - 0.75 * scored["std_delta_y"]

    regrets = []
    top3_hits = []
    for _, group in scored.groupby("base_id"):
        best_true = group["true_delta_y"].max()
        selected = group.sort_values("risk_adjusted_score", ascending=False).iloc[0]
        regrets.append(best_true - selected["true_delta_y"])
        top3_true_ids = set(group.sort_values("true_delta_y", ascending=False).head(3)["candidate_name"])
        top_pred = selected["candidate_name"]
        top3_hits.append(top_pred in top3_true_ids)

    return {
        "top1_regret": float(np.mean(regrets)),
        "top3_hit_rate": float(np.mean(top3_hits)),
    }
