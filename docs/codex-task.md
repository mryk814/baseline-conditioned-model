# Codex Task Prompt

Use this prompt when asking Codex to implement the prototype.

---

Please implement a Python prototype for this repository.

The project is a baseline-conditioned modeling demo for materials development. The goal is to predict and explain how a material property changes when composition or process parameters are perturbed from a baseline condition.

## Core problem

The main prediction target is:

```text
delta_y = y_new - y_base
```

where:

```text
new_x = base_x + delta_x
```

The explanation view should be locally linear:

```text
delta_y ≈ beta(base_x, organization, product_family)^T delta_x
```

## Please implement

### 1. Project setup

Create:

```text
requirements.txt
src/generate_data.py
src/features.py
src/models.py
src/local_linear.py
src/gp_models.py
src/evaluation.py
src/explanation.py
src/plotting.py
app/streamlit_app.py
notebooks/01_compare_models.ipynb
```

Use simple, readable Python. Prefer scikit-learn, pandas, numpy, matplotlib, and streamlit.

### 2. Synthetic data generation

Generate `data/materials_pairs.csv` and `data/candidate_actions.csv`.

The dataset should include:

- base material/process features
- delta features
- new features
- organization, equipment, product_family
- y_base, y_new, delta_y
- flags for OOD and sparse-direction cases

The synthetic function should include nonlinear effects, interactions, organization effects, and intentionally sparse change directions.

### 3. Models

Implement and compare:

1. Global Linear Absolute Model
2. Global Delta Linear Model
3. Local Linear Regression
4. Local Linear with Partial Pooling
5. GP Absolute Model
6. GP Delta Model

For GP models, scikit-learn `GaussianProcessRegressor` is enough for the MVP.

### 4. Local linear explanation

For local linear regression, return:

- prediction
- local coefficients
- local feature contributions
- nearest examples
- effective sample size
- direction coverage score
- same-organization weight ratio
- uncertainty estimate
- warning flags

Use ridge regularization for stability.

### 5. Evaluation

Compute:

- RMSE_delta_y
- MAE_delta_y
- R2_delta_y
- sign accuracy
- top1 regret for candidate ranking
- top3 hit rate
- uncertainty coverage if intervals are available
- error vs effective sample size
- error vs direction coverage

Use at least random split and one OOD split.

### 6. Visualizations

Create plots for:

- predicted vs true delta_y
- model comparison metrics
- uncertainty vs absolute error
- local contribution chart
- candidate ranking comparison

Use matplotlib.

### 7. Streamlit UI

Build a simple UI with:

- base condition selector
- delta input controls
- model selector
- prediction card
- local contribution chart
- nearest evidence table
- support metrics
- warning messages
- candidate comparison table

The UI should make it clear when a prediction is well supported and when it is extrapolating.

## Expected outcome

After running the prototype, a user should be able to answer:

1. Does delta prediction outperform absolute prediction for this setup?
2. Does local linear regression provide useful explanations?
3. Does GP provide better uncertainty estimates?
4. Does partial pooling help low-data organizations?
5. Can the system detect unsupported change directions?

## Notes

Keep the implementation compact. This is a research/demo prototype, not production code.

Use official documentation and primary sources for technical details. Do not rely on Qiita or Zenn as implementation references.
