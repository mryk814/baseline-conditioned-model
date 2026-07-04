from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONTINUOUS_FEATURES = [
    "C",
    "Mn",
    "Cr",
    "Ni",
    "Mo",
    "tempering_temp",
    "holding_time",
    "cooling_rate",
]

DELTA_FEATURES = [f"delta_{feature}" for feature in CONTINUOUS_FEATURES]
NEW_FEATURES = [f"new_{feature}" for feature in CONTINUOUS_FEATURES]
CATEGORICAL_FEATURES = ["organization", "equipment", "product_family"]


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]

    @property
    def all_columns(self) -> list[str]:
        return [*self.numeric, *self.categorical]


DELTA_MODEL_FEATURES = FeatureSpec(
    numeric=[*CONTINUOUS_FEATURES, *DELTA_FEATURES],
    categorical=CATEGORICAL_FEATURES,
)

ABSOLUTE_MODEL_FEATURES = FeatureSpec(
    numeric=CONTINUOUS_FEATURES,
    categorical=CATEGORICAL_FEATURES,
)


def make_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), spec.numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec.categorical),
        ]
    )


def delta_model_frame(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[DELTA_MODEL_FEATURES.all_columns].copy()


def absolute_state_frame(rows: pd.DataFrame, state: str) -> pd.DataFrame:
    if state not in {"base", "new"}:
        raise ValueError("state must be 'base' or 'new'")

    frame = rows[CATEGORICAL_FEATURES].copy()
    for feature in CONTINUOUS_FEATURES:
        frame[feature] = rows[feature] if state == "base" else rows[f"new_{feature}"]
    return frame[[*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES]]


def absolute_training_frame(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    base = absolute_state_frame(rows, "base")
    new = absolute_state_frame(rows, "new")
    x = pd.concat([base, new], ignore_index=True)
    y = pd.concat([rows["y_base"], rows["y_new"]], ignore_index=True)
    return x, y


def ensure_new_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    for feature in CONTINUOUS_FEATURES:
        out[f"new_{feature}"] = out[feature] + out[f"delta_{feature}"]
    return out
