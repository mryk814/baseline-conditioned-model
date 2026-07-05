from src.ui_text import (
    best_model_label,
    confidence_label,
    format_candidate_table,
    format_warning_message,
    model_comparison_table,
    model_tendency,
    risk_tone,
    top_decision_summary,
)


def test_warning_codes_are_translated_to_reader_facing_messages():
    assert "外挿" in format_warning_message("EXTRAPOLATION_RISK")
    assert "変更方向" in format_warning_message("LOW_DIRECTION_SUPPORT")
    assert "EXTRAPOLATION_RISK" not in format_warning_message("EXTRAPOLATION_RISK")


def test_confidence_and_risk_tone_follow_support_and_uncertainty():
    assert confidence_label(8.0, 0.82) == "高"
    assert confidence_label(18.0, 0.55) == "中"
    assert confidence_label(28.0, 0.25) == "低"
    assert risk_tone(["EXTRAPOLATION_RISK"]) == "高リスク"
    assert risk_tone(["LOW_DIRECTION_SUPPORT"]) == "要確認"
    assert risk_tone([]) == "根拠あり"


def test_top_decision_summary_is_plain_japanese():
    summary = top_decision_summary(
        pred_delta_y=27.6,
        std_delta_y=10.6,
        confidence="中",
        risk="要確認",
        strongest_driver="Mo",
    )

    assert "+27.6 MPa" in summary
    assert "信頼度は中" in summary
    assert "要確認" in summary
    assert "Mo" in summary


def test_candidate_table_uses_japanese_columns():
    rows = [
        {
            "candidate_name": "candidate_01",
            "pred_delta_y": 12.3,
            "std_delta_y": 4.5,
            "risk_adjusted_score": 8.9,
            "same_org_support": 0.72,
            "direction_support": 0.64,
            "warning_count": 1,
            "true_delta_y": 11.0,
        }
    ]

    table = format_candidate_table(rows)

    assert list(table.columns) == [
        "候補",
        "期待変化",
        "不確実性",
        "リスク調整後",
        "同組織の根拠",
        "方向サポート",
        "リスク数",
        "合成データ上の正解",
    ]
    assert table.iloc[0]["期待変化"] == "+12.3 MPa"


def test_model_comparison_table_summarizes_quality_and_tendency_in_japanese():
    metrics = {
        "Global delta linear": {
            "rmse_delta_y": 14.2,
            "mae_delta_y": 10.1,
            "sign_accuracy": 0.76,
            "interval_95_coverage": 0.94,
            "ood_detection_auroc": 0.62,
        },
        "Local partial pooling": {
            "rmse_delta_y": 9.8,
            "mae_delta_y": 7.4,
            "sign_accuracy": 0.84,
            "interval_95_coverage": 0.91,
            "ood_detection_auroc": 0.71,
        },
    }
    labels = {
        "Global delta linear": "全体・変化量線形",
        "Local partial pooling": "局所線形 + 部分プーリング",
    }

    table = model_comparison_table(metrics, labels)

    assert list(table.columns) == ["総合順位", "モデル", "予測誤差", "方向正解率", "不確実性カバー", "外挿検知", "傾向"]
    assert table.iloc[0]["モデル"] == "局所線形 + 部分プーリング"
    assert table.iloc[0]["予測誤差"] == "9.8 MPa"
    assert table.iloc[0]["方向正解率"] == "84%"
    assert "低データ条件" in table.iloc[0]["傾向"]
    assert best_model_label(metrics, labels, "sign_accuracy", higher_is_better=True) == "局所線形 + 部分プーリング"
    assert "不確実性" in model_tendency("GP delta")
