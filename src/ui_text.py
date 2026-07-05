from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


WARNING_MESSAGES = {
    "LOW_BASE_SUPPORT": "近いベース条件の事例が少ないため、根拠はやや薄めです。",
    "LOW_DIRECTION_SUPPORT": "この変更方向に近い過去事例が少なく、方向サポートが弱いです。",
    "HIGH_ORG_BORROWING": "同じ組織だけでは根拠が足りず、他組織の知見を多めに借りています。",
    "HIGH_MODEL_UNCERTAINTY": "モデルの不確実性が高く、予測幅を広めに見る必要があります。",
    "EXTRAPOLATION_RISK": "過去データの範囲から外挿気味です。実験前の確認を強くおすすめします。",
    "CONFLICTING_LOCAL_EVIDENCE": "近傍事例の改善/悪化方向が割れていて、局所根拠が一致していません。",
}


def format_warning_message(code: str) -> str:
    return WARNING_MESSAGES.get(code, f"追加確認が必要です: {code}")


def confidence_label(std_delta_y: float, direction_support: float) -> str:
    if std_delta_y <= 12 and direction_support >= 0.7:
        return "High"
    if std_delta_y <= 22 and direction_support >= 0.45:
        return "Medium"
    return "Low"


def risk_tone(warnings: Iterable[str]) -> str:
    warning_set = set(warnings)
    if {"EXTRAPOLATION_RISK", "HIGH_MODEL_UNCERTAINTY"} & warning_set:
        return "High risk"
    if warning_set:
        return "Needs review"
    return "Supported"


def top_decision_summary(
    pred_delta_y: float,
    std_delta_y: float,
    confidence: str,
    risk: str,
    strongest_driver: str,
) -> str:
    direction = "increase" if pred_delta_y >= 0 else "decrease"
    return (
        f"Expected {direction}: {pred_delta_y:+.1f} MPa "
        f"(uncertainty +/- {std_delta_y:.1f}). "
        f"Confidence is {confidence}; risk status is {risk}. "
        f"The largest local driver is {strongest_driver}."
    )


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.0%}"


def format_candidate_table(rows: pd.DataFrame | list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    table = pd.DataFrame(
        {
            "Candidate": frame["candidate_name"],
            "Expected change": frame["pred_delta_y"].map(lambda value: f"{value:+.1f} MPa"),
            "Uncertainty": frame["std_delta_y"].map(lambda value: f"+/- {value:.1f}"),
            "Risk-adjusted score": frame["risk_adjusted_score"].map(lambda value: f"{value:+.1f}"),
            "Same-org evidence": frame["same_org_support"].map(format_percent),
            "Direction support": frame["direction_support"].map(format_percent),
            "Risk flags": frame["warning_count"].map(lambda value: f"{int(value)}"),
            "Synthetic truth": frame["true_delta_y"].map(lambda value: f"{value:+.1f} MPa") if "true_delta_y" in frame else "-",
        }
    )
    return table
