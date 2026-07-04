# Decision UI Spec

The UI should demonstrate how the baseline-conditioned framing helps material-development decision making.

A simple Streamlit app is enough.

## 1. Base condition panel

Let the user select a baseline material/process condition.

Show:

- base composition
- base process condition
- organization
- equipment
- product family
- current property value

Example:

```text
Base: B-0017
Organization: Org A
Product family: plate
Strength: 810 MPa
```

## 2. Change input panel

Let the user enter a proposed perturbation.

Example:

```text
C +0.03
Mo +0.10
tempering_temp -20
```

Also allow selecting from pre-generated candidate actions.

## 3. Prediction card

Show the predicted change from baseline.

Example:

```text
Predicted delta strength: +35 MPa
Prediction interval: [+12, +58] MPa
Confidence: Medium
```

If multiple targets are available, show tradeoffs:

```text
Strength: +35 MPa
Toughness: -4 J
```

## 4. Local contribution chart

Show local linear contributions as a bar chart or waterfall chart.

Example:

```text
C increase: +18 MPa
Mo increase: +12 MPa
tempering_temp decrease: +8 MPa
Residual / interaction: -3 MPa
```

The chart should be based on local coefficients, not global coefficients.

## 5. Evidence panel

Show the data supporting the prediction.

Metrics:

- effective sample size
- nearest example count
- same-organization weight ratio
- average base distance
- average delta-direction similarity
- direction coverage score

Nearest example table:

```text
example_id
organization
product_family
base_distance
delta_similarity
observed_delta_y
weight
```

This panel is central to the concept. It lets the system say whether the prediction is grounded in similar past changes.

## 6. Warning panel

Generate warning flags and natural-language messages.

Possible flags:

- LOW_BASE_SUPPORT
- LOW_DIRECTION_SUPPORT
- HIGH_ORG_BORROWING
- HIGH_MODEL_UNCERTAINTY
- EXTRAPOLATION_RISK
- CONFLICTING_LOCAL_EVIDENCE

Example messages:

```text
⚠ This baseline condition is far from most training examples.
⚠ This change direction has few similar historical examples.
⚠ Most evidence comes from other organizations.
ℹ Similar product-family data is being borrowed.
```

## 7. Candidate comparison page

For a selected baseline, compare multiple proposed changes.

Columns:

```text
candidate_name
pred_delta_y
std_delta_y
risk_adjusted_score
same_org_support
direction_support
warning_count
```

Sort by either:

- expected improvement
- risk-adjusted score
- support / evidence quality

This page should show that the best decision is not always the highest predicted mean. A risky high-uncertainty candidate should be distinguishable from a well-supported moderate improvement.

## 8. Natural language explanation

Generate a concise explanation from model outputs.

Template:

```text
For this baseline, the model predicts {pred_delta_y} change in strength.
The largest local contributions are {top_contributions}.
This prediction mainly uses {effective_sample_size} effective nearby examples.
{same_org_ratio}% of the weight comes from the same organization.
The direction coverage is {direction_coverage}.
{warning_summary}
```

Example:

```text
For this baseline, the model predicts +35 MPa strength change.
The largest local contributions are C (+18 MPa), Mo (+12 MPa), and tempering temperature (+8 MPa).
This prediction mainly uses 18 effective nearby examples.
Only 32% of the weight comes from the same organization, so the model is borrowing evidence from similar product families in other organizations.
The C and Mo simultaneous change direction is sparse, so uncertainty is elevated.
```

## 9. What the UI should prove

The UI should make these points visible:

1. Delta prediction is closer to the user's actual decision question than absolute prediction.
2. Local coefficients are easier to explain than a black-box prediction.
3. GP-style uncertainty is useful, but should be paired with evidence diagnostics.
4. Organization-level partial pooling can be explained as controlled borrowing.
5. The system can say not only what it predicts, but when it has weak support.
