# Model Comparison Plan

The prototype should compare both predictive accuracy and decision usefulness.

## Model A: Global Linear Absolute Model

Predict the modified condition directly:

```text
y_new ~ new_x + organization + product_family
```

Use this as the simplest baseline.

Expected behavior:

- easy to implement
- may work for broad trends
- weak for local change explanations
- weak when effects depend on baseline condition

Outputs:

- pred_y_new
- pred_delta_y = pred_y_new - y_base

## Model B: Global Delta Linear Model

Predict the change directly:

```text
delta_y ~ base_x + delta_x + organization + product_family
```

This aligns better with the intended user question.

Outputs:

- pred_delta_y
- global coefficients
- feature contributions beta_j * delta_x_j

Expected behavior:

- interpretable
- good baseline for the delta framing
- may fail when local coefficients vary strongly

## Model C: Local Linear Regression

At prediction time, fit a weighted local model around the query.

```text
delta_y ~ delta_x
```

Weight each training row using:

```text
weight_i = base_similarity_i * delta_similarity_i * context_similarity_i
```

Use ridge regularization for stability.

Return:

- pred_delta_y
- local coefficients
- local contributions
- effective sample size
- nearest examples
- direction coverage score
- same-organization weight ratio
- uncertainty estimate
- warning flags

Expected behavior:

- strong explanation fit
- useful for evidence-based UI
- may be unstable when support is sparse

## Model D: Local Linear with Partial Pooling

Fit two local models:

- same-organization local model
- global local model

Mix predictions according to effective sample size:

```text
alpha = ess_same_org / (ess_same_org + kappa)
pred = alpha * pred_same_org + (1 - alpha) * pred_global
```

Return `alpha` as an explanation of how much organization-specific evidence was available.

Expected behavior:

- better for low-data organizations
- easier to explain than full multi-task GP
- useful bridge toward hierarchical modeling

## Model E: GP Absolute Model

Fit a Gaussian process for the absolute property:

```text
y = f(x)
```

Predict both base and modified condition:

```text
pred_delta_y = pred_y_new - pred_y_base
```

Return:

- pred_y_base
- pred_y_new
- pred_delta_y
- std_y_base
- std_y_new
- std_delta_y if feasible

Expected behavior:

- good uncertainty behavior
- may predict well
- explanation requires extra local sensitivity or neighbor evidence

## Model F: GP Delta Model

Fit a Gaussian process for the change directly:

```text
delta_y = h(base_x, delta_x, organization, product_family)
```

Return:

- pred_delta_y
- std_delta_y

Expected behavior:

- well aligned with the baseline-change question
- useful uncertainty
- still needs explanation layer for feature contributions

## Model G: GPX-inspired / Varying Coefficient Model

The desired long-term form is:

```text
delta_y ≈ beta(base_x, organization, product_family)^T delta_x
```

where the coefficient vector changes with the baseline and context.

For the prototype, approximate this by local linear regression plus bootstrap uncertainty. Later, replace with a proper varying-coefficient GP or GPX-style model.

## Metrics

### Prediction metrics

Focus on delta prediction.

- RMSE_delta_y
- MAE_delta_y
- R2_delta_y
- RMSE_y_new
- MAE_y_new
- R2_y_new

### Direction metrics

Decision makers often need to know whether a change improves or worsens a target.

- sign accuracy
- large-change sign accuracy

### Candidate ranking metrics

For each baseline, generate several candidate actions and evaluate whether the model chooses good ones.

- top1 regret
- top3 hit rate
- Spearman rank correlation
- risk-adjusted score quality

A simple risk-adjusted score:

```text
score = pred_delta_y - lambda * std_delta_y
```

### Uncertainty metrics

- 80% interval coverage
- 95% interval coverage
- mean interval width
- calibration curve
- negative log likelihood if available

### Applicability domain metrics

Use synthetic flags and distance-based diagnostics.

- OOD detection AUROC
- error vs effective sample size
- error vs direction coverage
- error vs nearest-neighbor distance
- error vs same-organization weight ratio

## Split strategy

Use multiple test splits.

1. Random split
2. Baseline OOD split
3. Delta-direction OOD split
4. Low-data organization split
5. Product-family holdout split

Random split alone is not enough. The important demo is whether the system can say when prediction should not be trusted.
