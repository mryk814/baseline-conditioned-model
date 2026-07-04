from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features import (
    ABSOLUTE_MODEL_FEATURES,
    DELTA_MODEL_FEATURES,
    absolute_state_frame,
    absolute_training_frame,
    delta_model_frame,
    make_preprocessor,
)


@dataclass
class PredictionResult:
    pred_delta_y: np.ndarray
    std_delta_y: np.ndarray
    pred_y_new: np.ndarray | None = None
    details: dict[str, object] = field(default_factory=dict)


class GlobalAbsoluteLinearModel:
    name = "Global Absolute Linear"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(ABSOLUTE_MODEL_FEATURES)),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        self.residual_std_ = 1.0

    def fit(self, rows: pd.DataFrame) -> "GlobalAbsoluteLinearModel":
        x, y = absolute_training_frame(rows)
        self.pipeline.fit(x, y)
        residuals = y.to_numpy() - self.pipeline.predict(x)
        self.residual_std_ = float(np.std(residuals, ddof=1))
        return self

    def predict(self, rows: pd.DataFrame) -> PredictionResult:
        base_pred = self.pipeline.predict(absolute_state_frame(rows, "base"))
        new_pred = self.pipeline.predict(absolute_state_frame(rows, "new"))
        pred_delta = new_pred - base_pred
        std = np.full(len(rows), np.sqrt(2.0) * max(self.residual_std_, 1e-6))
        return PredictionResult(pred_delta_y=pred_delta, pred_y_new=new_pred, std_delta_y=std)


class GlobalDeltaLinearModel:
    name = "Global Delta Linear"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(DELTA_MODEL_FEATURES)),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        self.residual_std_ = 1.0

    def fit(self, rows: pd.DataFrame) -> "GlobalDeltaLinearModel":
        x = delta_model_frame(rows)
        y = rows["delta_y"]
        self.pipeline.fit(x, y)
        residuals = y.to_numpy() - self.pipeline.predict(x)
        self.residual_std_ = float(np.std(residuals, ddof=1))
        return self

    def predict(self, rows: pd.DataFrame) -> PredictionResult:
        pred_delta = self.pipeline.predict(delta_model_frame(rows))
        std = np.full(len(rows), max(self.residual_std_, 1e-6))
        return PredictionResult(
            pred_delta_y=pred_delta,
            pred_y_new=rows["y_base"].to_numpy() + pred_delta,
            std_delta_y=std,
        )
