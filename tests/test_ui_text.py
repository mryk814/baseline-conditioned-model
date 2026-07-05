from src.ui_text import (
    confidence_label,
    format_candidate_table,
    format_warning_message,
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
