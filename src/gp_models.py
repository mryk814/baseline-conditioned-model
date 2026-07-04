from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.pipeline import Pipeline

from src.features import (
    ABSOLUTE_MODEL_FEATURES,
    DELTA_MODEL_FEATURES,
    absolute_state_frame,
    absolute_training_frame,
    delta_model_frame,
    make_preprocessor,
)
from src.models import PredictionResult


def _kernel():
    return ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=1.0, length_scale_bounds="fixed"
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds="fixed")


class GPAbsoluteModel:
    name = "GP Absolute Model"

    def __init__(self, max_train_size: int = 350, random_state: int | None = 42):
        self.max_train_size = max_train_size
        self.random_state = random_state

    def fit(self, rows: pd.DataFrame) -> "GPAbsoluteModel":
        x, y = absolute_training_frame(rows)
        if len(x) > self.max_train_size:
            sample = x.sample(self.max_train_size, random_state=self.random_state).index
            x = x.loc[sample]
            y = y.loc[sample]
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(ABSOLUTE_MODEL_FEATURES)),
                ("model", GaussianProcessRegressor(kernel=_kernel(), normalize_y=True, random_state=self.random_state)),
            ]
        )
        self.pipeline.fit(x, y)
        return self

    def predict(self, rows: pd.DataFrame) -> PredictionResult:
        base_x = absolute_state_frame(rows, "base")
        new_x = absolute_state_frame(rows, "new")
        model = self.pipeline.named_steps["model"]
        base_transformed = self.pipeline.named_steps["preprocess"].transform(base_x)
        new_transformed = self.pipeline.named_steps["preprocess"].transform(new_x)
        pred_base, std_base = model.predict(base_transformed, return_std=True)
        pred_new, std_new = model.predict(new_transformed, return_std=True)
        return PredictionResult(
            pred_delta_y=pred_new - pred_base,
            pred_y_new=pred_new,
            std_delta_y=np.sqrt(np.square(std_base) + np.square(std_new)),
            details={
                "pred_y_base": pred_base,
                "std_y_base": std_base,
                "std_y_new": std_new,
            },
        )


class GPDeltaModel:
    name = "GP Delta Model"

    def __init__(self, max_train_size: int = 350, random_state: int | None = 42):
        self.max_train_size = max_train_size
        self.random_state = random_state

    def fit(self, rows: pd.DataFrame) -> "GPDeltaModel":
        x = delta_model_frame(rows)
        y = rows["delta_y"]
        if len(x) > self.max_train_size:
            sample = x.sample(self.max_train_size, random_state=self.random_state).index
            x = x.loc[sample]
            y = y.loc[sample]
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(DELTA_MODEL_FEATURES)),
                ("model", GaussianProcessRegressor(kernel=_kernel(), normalize_y=True, random_state=self.random_state)),
            ]
        )
        self.pipeline.fit(x, y)
        return self

    def predict(self, rows: pd.DataFrame) -> PredictionResult:
        x = delta_model_frame(rows)
        model = self.pipeline.named_steps["model"]
        transformed = self.pipeline.named_steps["preprocess"].transform(x)
        pred, std = model.predict(transformed, return_std=True)
        return PredictionResult(
            pred_delta_y=pred,
            pred_y_new=rows["y_base"].to_numpy() + pred,
            std_delta_y=std,
        )
