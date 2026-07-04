# Research Map

This document maps the current idea to nearby existing methods so that we avoid reinventing the wheel.

## 1. Local linear regression

Closest to the intuitive user-facing explanation.

At a query baseline, fit a small linear model around nearby historical examples:

```text
delta_y ~ delta_x
```

Useful outputs:

- local coefficients
- local feature contributions
- nearest examples
- support / coverage scores

Main weakness:

- uncertainty and pooling need extra design
- can be unstable in sparse regions

## 2. Locally weighted regression / LOESS-style thinking

Similar to local linear regression, but with explicit distance-based weights.

Good for:

- intuitive explanations
- showing the historical neighborhood
- saying whether the query is interpolation or extrapolation

## 3. Gaussian process regression

GP is similar because it uses kernels and naturally handles uncertainty.

Standard form:

```text
y = f(x)
```

For this project, an absolute GP can be used as:

```text
delta_y = f(base_x + delta_x) - f(base_x)
```

Good for:

- prediction in small-data regimes
- predictive uncertainty
- smooth nonlinear response surfaces

Weakness for this project:

- the explanation is naturally about point prediction, not about a user-facing change narrative
- local feature contribution requires an additional explanation layer

## 4. GP delta model

Instead of modeling absolute property values, model the change directly:

```text
delta_y = h(base_x, delta_x, context)
```

This is closer to the intended product question.

Good for:

- baseline-conditioned decisions
- uncertainty over changes
- candidate ranking

Weakness:

- less direct if the original data is not naturally paired
- still needs explanation and support diagnostics

## 5. GPX / Gaussian Process Regression with Local Explanation

Very relevant prior art.

The idea is to combine Gaussian-process prediction with local linear explanations. This is close to the desired product pattern:

```text
predict with GP-style machinery
explain with local linear coefficients
```

Potential use here:

- use GPX as inspiration for local weights / local coefficients
- adapt it from absolute `y` prediction to baseline-conditioned `delta_y` prediction

## 6. Varying coefficient model

A clean formulation for the long-term model:

```text
delta_y ≈ beta(base_x, context)^T delta_x
```

The coefficient vector itself changes with the baseline and context.

This maps well to materials intuition:

- the effect of increasing C depends on the current composition and process
- the effect of changing temperature depends on the baseline condition
- organization or equipment may shift the local coefficient

Potential implementation path:

1. start with local linear regression
2. estimate local coefficients
3. later model coefficients as smooth functions of base condition
4. eventually use GP priors or hierarchical Bayesian modeling

## 7. Partial pooling / hierarchical modeling

Needed for organization, equipment, and product-family differences.

Problem:

- full pooling ignores local practices and measurement differences
- no pooling wastes data and fails for low-data groups

Practical MVP:

```text
pred = alpha * pred_same_org + (1 - alpha) * pred_global
alpha = ess_same_org / (ess_same_org + kappa)
```

Longer-term options:

- hierarchical Bayesian regression
- multi-task Gaussian process
- latent-variable GP for categorical sources
- source-aware kernels

## 8. Applicability domain / OOD detection

Essential for decision support.

The system should not only predict. It should also say whether it has enough evidence.

Useful diagnostics:

- base distance to nearest examples
- delta direction similarity
- effective sample size
- same-organization weight ratio
- local residual variance
- GP predictive uncertainty
- sparse-direction flags

Example message:

```text
This change direction is poorly represented in historical data, so the prediction is high risk.
```

## 9. Bayesian optimization

Not necessary for the first prototype, but relevant later.

Once prediction and uncertainty are available, candidate suggestions can use:

- expected improvement
- upper confidence bound
- risk-adjusted improvement
- constrained optimization for multiple properties

## Suggested MVP positioning

The MVP should not claim to invent a new algorithm.

Position it as:

> A baseline-conditioned material-change prediction demo that combines delta modeling, local linear explanations, GP uncertainty, partial pooling, and applicability-domain diagnostics.
