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

MODEL_TENDENCIES = {
    "Global absolute linear": "絶対値を直接見る基準モデル。大まかな水準差の確認向きです。",
    "Global delta linear": "変化量に素直な基準モデル。平均的な方向性を見やすいです。",
    "Local linear": "近い事例から説明しやすいモデル。事例が薄い領域では注意が必要です。",
    "Local partial pooling": "近傍事例と全体傾向を混ぜるモデル。低データ条件でも崩れにくいです。",
    "GP absolute": "絶対値を不確実性込みで見るモデル。予測幅の大きさを確認しやすいです。",
    "GP delta": "変化量を不確実性込みで見るモデル。改善方向と予測幅を合わせて見やすいです。",
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


def model_tendency(model_key: str) -> str:
    return MODEL_TENDENCIES.get(model_key, "このモデルの傾向は評価結果から確認してください。")


def best_model_label(
    metrics: dict[str, dict[str, float]],
    model_labels: dict[str, str],
    metric: str,
    *,
    higher_is_better: bool = False,
) -> str:
    frame = pd.DataFrame(metrics).T
    values = frame[metric].dropna()
    if values.empty:
        return "-"
    best_key = values.idxmax() if higher_is_better else values.idxmin()
    return model_labels.get(best_key, best_key)


def model_comparison_table(metrics: dict[str, dict[str, float]], model_labels: dict[str, str]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics).T.copy()
    frame = frame.sort_values(["rmse_delta_y", "mae_delta_y", "sign_accuracy"], ascending=[True, True, False])

    rows = []
    for rank, (model_key, row) in enumerate(frame.iterrows(), start=1):
        rows.append(
            {
                "総合順位": rank,
                "モデル": model_labels.get(model_key, model_key),
                "予測誤差": f"{row['rmse_delta_y']:.1f} MPa",
                "方向正解率": format_percent(row.get("sign_accuracy")),
                "不確実性カバー": format_percent(row.get("interval_95_coverage")),
                "外挿検知": format_percent(row.get("ood_detection_auroc")),
                "傾向": model_tendency(model_key),
            }
        )
    return pd.DataFrame(rows)


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
