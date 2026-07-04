from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import CONTINUOUS_FEATURES, DELTA_FEATURES, ensure_new_features


ORGANIZATIONS = ["Org A", "Org B", "Org C", "Org D"]
EQUIPMENT = ["furnace_1", "furnace_2", "line_3"]
PRODUCT_FAMILIES = ["plate", "bar", "wire"]


def _rng(random_state: int | None) -> np.random.Generator:
    return np.random.default_rng(random_state)


def true_strength(frame: pd.DataFrame) -> np.ndarray:
    c = frame["C"].to_numpy()
    mn = frame["Mn"].to_numpy()
    cr = frame["Cr"].to_numpy()
    ni = frame["Ni"].to_numpy()
    mo = frame["Mo"].to_numpy()
    temp = frame["tempering_temp"].to_numpy()
    hold = frame["holding_time"].to_numpy()
    cooling = frame["cooling_rate"].to_numpy()

    org_effect = frame["organization"].map({"Org A": 18, "Org B": -8, "Org C": 5, "Org D": -22}).to_numpy()
    equipment_effect = frame["equipment"].map({"furnace_1": 6, "furnace_2": -4, "line_3": 0}).to_numpy()
    family_effect = frame["product_family"].map({"plate": 22, "bar": 5, "wire": -18}).to_numpy()

    return (
        520
        + 420 * c
        + 55 * mn
        + 36 * cr
        + 25 * ni
        + 115 * mo
        - 0.32 * temp
        + 12 * np.log1p(hold)
        + 8.5 * np.sqrt(cooling)
        + 210 * c * mo
        - 0.035 * (temp - 560) * mo
        + 24 * np.sin(cr + ni)
        + org_effect
        + equipment_effect
        + family_effect
    )


def _base_conditions(n_baselines: int, rng: np.random.Generator) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "base_id": [f"B-{i:04d}" for i in range(n_baselines)],
            "C": rng.uniform(0.12, 0.48, n_baselines),
            "Mn": rng.uniform(0.4, 1.8, n_baselines),
            "Cr": rng.uniform(0.1, 1.6, n_baselines),
            "Ni": rng.uniform(0.0, 1.2, n_baselines),
            "Mo": rng.uniform(0.0, 0.45, n_baselines),
            "tempering_temp": rng.uniform(470, 680, n_baselines),
            "holding_time": rng.uniform(0.5, 6.0, n_baselines),
            "cooling_rate": rng.uniform(0.4, 9.5, n_baselines),
            "organization": rng.choice(ORGANIZATIONS, n_baselines, p=[0.38, 0.28, 0.24, 0.10]),
            "equipment": rng.choice(EQUIPMENT, n_baselines),
            "product_family": rng.choice(PRODUCT_FAMILIES, n_baselines, p=[0.45, 0.35, 0.20]),
        }
    )
    base["y_base"] = true_strength(base) + rng.normal(0, 6.0, n_baselines)
    return base


def _sample_delta(index: int, rng: np.random.Generator) -> dict[str, float]:
    delta = {
        "delta_C": rng.normal(0.0, 0.018),
        "delta_Mn": rng.normal(0.0, 0.09),
        "delta_Cr": rng.normal(0.0, 0.08),
        "delta_Ni": rng.normal(0.0, 0.06),
        "delta_Mo": rng.normal(0.0, 0.035),
        "delta_tempering_temp": rng.normal(0.0, 18.0),
        "delta_holding_time": rng.normal(0.0, 0.45),
        "delta_cooling_rate": rng.normal(0.0, 0.8),
    }

    if index % 17 == 0:
        delta["delta_C"] = rng.choice([0.035, -0.035])
        delta["delta_Mo"] = rng.choice([0.08, -0.08])
    if index % 29 == 0:
        delta["delta_tempering_temp"] = rng.choice([70.0, -70.0])
    return delta


def _clip_new_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = ensure_new_features(frame)
    bounds = {
        "C": (0.05, 0.65),
        "Mn": (0.1, 2.2),
        "Cr": (0.0, 2.1),
        "Ni": (0.0, 1.8),
        "Mo": (0.0, 0.7),
        "tempering_temp": (390, 760),
        "holding_time": (0.1, 8.0),
        "cooling_rate": (0.1, 12.0),
    }
    for feature, (low, high) in bounds.items():
        out[f"new_{feature}"] = out[f"new_{feature}"].clip(low, high)
        out[f"delta_{feature}"] = out[f"new_{feature}"] - out[feature]
    return out


def _add_targets_and_flags(frame: pd.DataFrame, rng: np.random.Generator, target_column: str = "delta_y") -> pd.DataFrame:
    out = _clip_new_values(frame)
    new_state = out[["base_id", "organization", "equipment", "product_family", *[f"new_{f}" for f in CONTINUOUS_FEATURES]]].copy()
    new_state = new_state.rename(columns={f"new_{feature}": feature for feature in CONTINUOUS_FEATURES})
    y_new = true_strength(new_state) + rng.normal(0, 6.0, len(out))
    out["y_new"] = y_new
    out[target_column] = out["y_new"] - out["y_base"]
    out["delta_y"] = out[target_column]

    delta_norm = np.sqrt(np.sum(np.square(out[DELTA_FEATURES].to_numpy()), axis=1))
    out["is_sparse_direction"] = (
        (out["delta_C"].abs() > 0.028) & (out["delta_Mo"].abs() > 0.065)
    ) | (out["delta_tempering_temp"].abs() > 55)
    out["is_ood"] = (delta_norm > np.quantile(delta_norm, 0.88)) | (out["new_tempering_temp"] > 720)
    return out


def generate_material_pairs(
    n_baselines: int = 120,
    actions_per_base: int = 5,
    random_state: int | None = 42,
) -> pd.DataFrame:
    rng = _rng(random_state)
    base = _base_conditions(n_baselines, rng)
    rows: list[dict[str, object]] = []
    for _, base_row in base.iterrows():
        for action_index in range(actions_per_base):
            row = base_row.to_dict()
            row["pair_id"] = f"{base_row['base_id']}-P{action_index:02d}"
            row.update(_sample_delta(len(rows), rng))
            rows.append(row)

    pairs = pd.DataFrame(rows)
    return _add_targets_and_flags(pairs, rng)


def generate_candidate_actions(
    pairs: pd.DataFrame,
    candidates_per_base: int = 6,
    n_baselines: int = 20,
    random_state: int | None = 42,
) -> pd.DataFrame:
    rng = _rng(random_state)
    base_rows = pairs.drop_duplicates("base_id").head(n_baselines)
    rows: list[dict[str, object]] = []
    for _, base_row in base_rows.iterrows():
        for candidate_index in range(candidates_per_base):
            row = {column: base_row[column] for column in ["base_id", "organization", "equipment", "product_family", "y_base", *CONTINUOUS_FEATURES]}
            row["candidate_name"] = f"candidate_{candidate_index + 1:02d}"
            row["pair_id"] = f"{base_row['base_id']}-C{candidate_index:02d}"
            row.update(_sample_delta(candidate_index + len(rows), rng))
            if candidate_index == 0:
                row["delta_C"] = 0.03
                row["delta_Mo"] = 0.08
            rows.append(row)

    candidates = _add_targets_and_flags(pd.DataFrame(rows), rng, target_column="true_delta_y")
    return candidates


def save_demo_data(data_dir: str | Path = "data", random_state: int | None = 42) -> tuple[Path, Path]:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    pairs = generate_material_pairs(random_state=random_state)
    candidates = generate_candidate_actions(pairs, random_state=random_state)
    pairs_path = data_path / "materials_pairs.csv"
    candidates_path = data_path / "candidate_actions.csv"
    pairs.to_csv(pairs_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    return pairs_path, candidates_path


if __name__ == "__main__":
    pairs_file, candidates_file = save_demo_data()
    print(f"Wrote {pairs_file}")
    print(f"Wrote {candidates_file}")
