# Implementation Brief

## Goal

Build a prototype for material-development decision support.

The core question is:

> Given a baseline material/process condition, what happens if we perturb composition or process parameters?

The prototype should compare predictive performance and show what decision-relevant information can be surfaced to users.

The desired modeling frame is:

```text
delta_y = f(base_x + delta_x) - f(base_x)
```

For explanation, expose a local linear view:

```text
delta_y ≈ beta(base_x, organization, product_family)^T delta_x
```

This lets the UI explain:

- expected property change from the baseline
- local feature contributions
- nearby historical evidence
- whether the change direction is well covered
- how much knowledge is borrowed from other organizations or product families
- uncertainty and extrapolation risk

## Synthetic data

Create paired rows. Each row represents one baseline condition and one modified condition.

Continuous base features:

- C
- Mn
- Cr
- Ni
- Mo
- tempering_temp
- holding_time
- cooling_rate

Categorical context features:

- organization
- equipment
- product_family

For each continuous feature, create a matching delta feature and a new-value feature.

Targets:

- y_base
- y_new
- delta_y = y_new - y_base

Start with strength as the main target. Optionally add toughness later.

## Data-generating behavior

The synthetic function should include:

- nonlinear effects
- feature interactions
- local coefficient variation
- organization / equipment effects
- sparse regions and missing change directions

The purpose is to create a dataset where a global linear model is understandable but insufficient, while local linear and GP-style models have a meaningful advantage.

## Output files

Generate:

- data/materials_pairs.csv
- data/candidate_actions.csv

`materials_pairs.csv` should include IDs, base features, delta features, new features, categorical context, targets, and flags for OOD or sparse-direction cases.

`candidate_actions.csv` should include multiple candidate changes for selected base conditions so that ranking and decision metrics can be evaluated.

## Implementation phases

### Phase 1

- synthetic data generation
- train/test split
- global delta linear model
- local linear regression
- GP absolute model
- GP delta model
- basic metrics and plots

### Phase 2

- candidate action generation
- top-k candidate ranking
- regret metrics
- sparse-direction / OOD warnings

### Phase 3

- Streamlit UI
- base condition selection
- proposed delta input
- predicted delta_y
- prediction interval / uncertainty
- local feature contributions
- nearest historical examples
- support / coverage scores
- warnings

### Phase 4

- partial pooling by organization
- same-organization local model
- global local model
- effective-sample-size-based shrinkage
- borrowing ratio in the UI

## Reference policy

Use official docs, papers, official repositories, and primary sources where possible. Do not use Qiita or Zenn as implementation references.
