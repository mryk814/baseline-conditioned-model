from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


WARNING_MESSAGES = {
    "LOW_BASE_SUPPORT": "nearby baseline support is thin",
    "LOW_DIRECTION_SUPPORT": "similar historical change directions are sparse",
    "HIGH_ORG_BORROWING": "the estimate borrows substantial evidence from other organizations",
    "HIGH_MODEL_UNCERTAINTY": "model uncertainty is elevated",
    "EXTRAPOLATION_RISK": "the query is outside well-supported historical regions",
    "CONFLICTING_LOCAL_EVIDENCE": "nearby evidence has conflicting improvement directions",
}


@dataclass
class TextExplanation:
    text: str
    warnings: list[str]
    details: object


def build_prediction_explanation(model: object, query: pd.Series) -> TextExplanation:
    if hasattr(model, "explain_one"):
        details = model.explain_one(query)
        contributions = sorted(details.contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        contribution_text = ", ".join(f"{name.replace('delta_', '')} {value:+.1f}" for name, value in contributions)
        warning_text = "; ".join(WARNING_MESSAGES.get(flag, flag) for flag in details.warnings) or "support diagnostics look acceptable"
        text = (
            f"The model predicts {details.pred_delta_y:+.1f} MPa change. "
            f"Largest local contributions are {contribution_text}. "
            f"It uses {details.effective_sample_size:.1f} effective nearby examples, "
            f"with {details.same_organization_weight_ratio:.0%} same-organization support. "
            f"Direction coverage is {details.direction_coverage:.0%}; "
            f"average base distance is {details.average_base_distance:.2f}; {warning_text}."
        )
        return TextExplanation(text=text, warnings=details.warnings, details=details)

    prediction = model.predict(pd.DataFrame([query]))
    text = f"The model predicts {prediction.pred_delta_y[0]:+.1f} MPa change with estimated std {prediction.std_delta_y[0]:.1f}."
    return TextExplanation(text=text, warnings=[], details=prediction)
