from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features import CONTINUOUS_FEATURES, DELTA_FEATURES
from src.models import PredictionResult


@dataclass
class LocalExplanation:
    pred_delta_y: float
    std_delta_y: float
    intercept: float
    coefficients: dict[str, float]
    contributions: dict[str, float]
    nearest_examples: pd.DataFrame
    effective_sample_size: float
    direction_coverage: float
    same_organization_weight_ratio: float
    average_base_distance: float
    average_delta_similarity: float
    warnings: list[str] = field(default_factory=list)


class LocalLinearRegressor:
    name = "Local Linear Regression"

    def __init__(self, k_neighbors: int = 60, ridge_alpha: float = 2.0, base_bandwidth: float = 1.4):
        self.k_neighbors = k_neighbors
        self.ridge_alpha = ridge_alpha
        self.base_bandwidth = base_bandwidth
        self.delta_features = DELTA_FEATURES
        self.base_features = CONTINUOUS_FEATURES

    def fit(self, rows: pd.DataFrame) -> "LocalLinearRegressor":
        self.train_ = rows.reset_index(drop=True).copy()
        self.base_scale_ = self.train_[self.base_features].std().replace(0, 1.0)
        self.delta_scale_ = self.train_[self.delta_features].std().replace(0, 1.0)
        return self

    def predict(self, rows: pd.DataFrame) -> PredictionResult:
        explanations = [self.explain_one(row) for _, row in rows.iterrows()]
        return PredictionResult(
            pred_delta_y=np.array([item.pred_delta_y for item in explanations]),
            std_delta_y=np.array([item.std_delta_y for item in explanations]),
            pred_y_new=rows["y_base"].to_numpy() + np.array([item.pred_delta_y for item in explanations]),
            details={"explanations": explanations},
        )

    def explain_one(self, query: pd.Series) -> LocalExplanation:
        return self._explain_with_frame(self.train_, query)

    def _weights(self, frame: pd.DataFrame, query: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_values = frame[self.base_features].to_numpy(dtype=float)
        query_base = query[self.base_features].to_numpy(dtype=float)
        base_scale = self.base_scale_.to_numpy(dtype=float)
        base_diff = (base_values - query_base) / base_scale
        base_distance = np.sqrt(np.square(base_diff).mean(axis=1))
        base_similarity = np.exp(-0.5 * np.square(base_distance / self.base_bandwidth))

        train_delta = frame[self.delta_features].to_numpy(dtype=float) / self.delta_scale_.to_numpy()
        query_delta = query[self.delta_features].to_numpy(dtype=float) / self.delta_scale_.to_numpy()
        query_norm = np.linalg.norm(query_delta)
        train_norm = np.linalg.norm(train_delta, axis=1)
        if query_norm < 1e-9:
            cosine = np.ones(len(frame))
        else:
            cosine = (train_delta @ query_delta) / np.maximum(train_norm * query_norm, 1e-9)
        direction_similarity = np.clip((cosine + 1.0) / 2.0, 0.02, 1.0)

        context_similarity = np.full(len(frame), 0.35)
        context_similarity += 0.45 * (frame["organization"].to_numpy() == query["organization"])
        context_similarity += 0.15 * (frame["product_family"].to_numpy() == query["product_family"])
        context_similarity += 0.05 * (frame["equipment"].to_numpy() == query["equipment"])

        weights = base_similarity * direction_similarity * context_similarity
        return weights, cosine, base_distance

    def _explain_with_frame(self, frame: pd.DataFrame, query: pd.Series) -> LocalExplanation:
        weights, cosine, base_distance = self._weights(frame, query)
        order = np.argsort(weights)[::-1][: min(self.k_neighbors, len(frame))]
        local = frame.iloc[order].copy()
        local_weights = np.maximum(weights[order], 1e-8)

        x = local[self.delta_features].to_numpy(dtype=float)
        y = local["delta_y"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(local)), x])
        penalty = np.eye(design.shape[1]) * self.ridge_alpha
        penalty[0, 0] = 0.0
        weighted_design = design * local_weights[:, None]
        beta = np.linalg.solve(design.T @ weighted_design + penalty, weighted_design.T @ y)

        query_delta = query[self.delta_features].to_numpy(dtype=float)
        pred = float(beta[0] + query_delta @ beta[1:])
        residuals = y - design @ beta
        weight_sum = float(local_weights.sum())
        residual_var = float(np.average(np.square(residuals), weights=local_weights)) if weight_sum else 0.0
        ess = float(weight_sum**2 / np.maximum(np.square(local_weights).sum(), 1e-9))
        std = float(np.sqrt(max(residual_var, 1.0)) * np.sqrt(1.0 + 1.0 / max(ess, 1.0)))
        direction_coverage = float(np.average(np.clip(cosine[order], 0, 1), weights=local_weights))
        same_org_ratio = float(local_weights[local["organization"].to_numpy() == query["organization"]].sum() / max(weight_sum, 1e-9))
        average_base_distance = float(np.average(base_distance[order], weights=local_weights))
        average_delta_similarity = float(np.average(cosine[order], weights=local_weights))

        nearest = local[
            ["pair_id", "organization", "product_family", "delta_y", *self.delta_features]
        ].head(10).copy()
        nearest = nearest.rename(columns={"delta_y": "observed_delta_y"})
        nearest["weight"] = local_weights[: len(nearest)]
        nearest["delta_similarity"] = cosine[order][: len(nearest)]
        nearest["base_distance"] = base_distance[order][: len(nearest)]

        warnings = []
        if ess < 12:
            warnings.append("LOW_BASE_SUPPORT")
        if direction_coverage < 0.45:
            warnings.append("LOW_DIRECTION_SUPPORT")
        if same_org_ratio < 0.35:
            warnings.append("HIGH_ORG_BORROWING")
        if std > 35:
            warnings.append("HIGH_MODEL_UNCERTAINTY")
        if average_base_distance > 1.7 or bool(query.get("is_ood", False)) or bool(query.get("is_sparse_direction", False)):
            warnings.append("EXTRAPOLATION_RISK")
        weighted_positive = float(local_weights[y > 0].sum() / max(weight_sum, 1e-9))
        if 0.35 < weighted_positive < 0.65 and abs(pred) < std:
            warnings.append("CONFLICTING_LOCAL_EVIDENCE")

        coefficients = dict(zip(self.delta_features, beta[1:]))
        contributions = {feature: float(coefficients[feature] * query[feature]) for feature in self.delta_features}
        return LocalExplanation(
            pred_delta_y=pred,
            std_delta_y=std,
            intercept=float(beta[0]),
            coefficients={key: float(value) for key, value in coefficients.items()},
            contributions=contributions,
            nearest_examples=nearest,
            effective_sample_size=ess,
            direction_coverage=direction_coverage,
            same_organization_weight_ratio=same_org_ratio,
            average_base_distance=average_base_distance,
            average_delta_similarity=average_delta_similarity,
            warnings=warnings,
        )


class PartialPoolingLocalLinearRegressor(LocalLinearRegressor):
    name = "Local Linear with Partial Pooling"

    def __init__(self, k_neighbors: int = 60, ridge_alpha: float = 2.0, base_bandwidth: float = 1.4, kappa: float = 18.0):
        super().__init__(k_neighbors=k_neighbors, ridge_alpha=ridge_alpha, base_bandwidth=base_bandwidth)
        self.kappa = kappa

    def explain_one(self, query: pd.Series) -> LocalExplanation:
        global_explanation = self._explain_with_frame(self.train_, query)
        same_org = self.train_[self.train_["organization"] == query["organization"]]
        if len(same_org) < 5:
            return global_explanation

        same_explanation = self._explain_with_frame(same_org, query)
        alpha = same_explanation.effective_sample_size / (same_explanation.effective_sample_size + self.kappa)
        pred = alpha * same_explanation.pred_delta_y + (1.0 - alpha) * global_explanation.pred_delta_y
        std = alpha * same_explanation.std_delta_y + (1.0 - alpha) * global_explanation.std_delta_y

        merged = global_explanation
        merged.pred_delta_y = float(pred)
        merged.std_delta_y = float(std)
        merged.same_organization_weight_ratio = float(alpha)
        merged.warnings = sorted(set(global_explanation.warnings + same_explanation.warnings))
        if alpha < 0.35 and "HIGH_ORG_BORROWING" not in merged.warnings:
            merged.warnings.append("HIGH_ORG_BORROWING")
        return merged
