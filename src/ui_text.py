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
        return "高"
    if std_delta_y <= 22 and direction_support >= 0.45:
        return "中"
    return "低"


def risk_tone(warnings: Iterable[str]) -> str:
    warning_set = set(warnings)
    if {"EXTRAPOLATION_RISK", "HIGH_MODEL_UNCERTAINTY"} & warning_set:
        return "高リスク"
    if warning_set:
        return "要確認"
    return "根拠あり"


def top_decision_summary(
    pred_delta_y: float,
    std_delta_y: float,
    confidence: str,
    risk: str,
    strongest_driver: str,
) -> str:
    direction = "上がる" if pred_delta_y >= 0 else "下がる"
    return (
        f"この変更では強度が {pred_delta_y:+.1f} MPa {direction}見込みです。"
        f"不確実性は +/- {std_delta_y:.1f}、信頼度は{confidence}、リスク判定は{risk}です。"
        f"局所的に最も効いている要因は {strongest_driver} です。"
    )


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.0%}"


def format_candidate_table(rows: pd.DataFrame | list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    table = pd.DataFrame(
        {
            "候補": frame["candidate_name"],
            "期待変化": frame["pred_delta_y"].map(lambda value: f"{value:+.1f} MPa"),
            "不確実性": frame["std_delta_y"].map(lambda value: f"+/- {value:.1f}"),
            "リスク調整後": frame["risk_adjusted_score"].map(lambda value: f"{value:+.1f}"),
            "同組織の根拠": frame["same_org_support"].map(format_percent),
            "方向サポート": frame["direction_support"].map(format_percent),
            "リスク数": frame["warning_count"].map(lambda value: f"{int(value)}"),
            "合成データ上の正解": frame["true_delta_y"].map(lambda value: f"{value:+.1f} MPa") if "true_delta_y" in frame else "-",
        }
    )
    return table
