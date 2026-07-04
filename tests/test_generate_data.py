from src.generate_data import generate_candidate_actions, generate_material_pairs
from src.features import CONTINUOUS_FEATURES


def test_generate_material_pairs_contains_baseline_delta_new_and_targets():
    data = generate_material_pairs(n_baselines=12, actions_per_base=3, random_state=7)

    assert len(data) == 36
    for feature in CONTINUOUS_FEATURES:
        assert feature in data.columns
        assert f"delta_{feature}" in data.columns
        assert f"new_{feature}" in data.columns

    assert {"base_id", "pair_id", "organization", "equipment", "product_family"}.issubset(data.columns)
    assert {"y_base", "y_new", "delta_y", "is_ood", "is_sparse_direction"}.issubset(data.columns)
    assert ((data["y_new"] - data["y_base"] - data["delta_y"]).abs() < 1e-8).all()
    assert data["is_sparse_direction"].any()


def test_generate_candidate_actions_reuses_known_baselines_and_names_candidates():
    pairs = generate_material_pairs(n_baselines=10, actions_per_base=2, random_state=12)
    candidates = generate_candidate_actions(pairs, candidates_per_base=4, random_state=12)

    assert set(candidates["base_id"]).issubset(set(pairs["base_id"]))
    assert candidates.groupby("base_id").size().min() == 4
    assert candidates["candidate_name"].str.startswith("candidate_").all()
    assert {"true_delta_y", "is_ood", "is_sparse_direction"}.issubset(candidates.columns)
