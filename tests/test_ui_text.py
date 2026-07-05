from src.ui_text import (
    confidence_label,
    format_candidate_table,
    format_warning_message,
    risk_tone,
    top_decision_summary,
)


def test_warning_codes_are_translated_to_reader_facing_messages():
    assert "外挿" in format_warning_message("EXTRAPOLATION_RISK")
    assert "方向" in format_warning_message("LOW_DIRECTION_SUPPORT")
    assert "EXTRAPOLATION_RISK" not in format_warning_message("EXTRAPOLATION_RISK")


def test_confidence_and_risk_tone_follow_support_and_uncertainty():
    assert confidence_label(8.0, 0.82) == "High"
    assert confidence_label(18.0, 0.55) == "Medium"
    assert confidence_label(28.0, 0.25) == "Low"
    assert risk_tone(["EXTRAPOLATION_RISK"]) == "High risk"
    assert risk_tone(["LOW_DIRECTION_SUPPORT"]) == "Needs review"
    assert risk_tone([]) == "Supported"


def test_top_decision_summary_is_plain_language():
    summary = top_decision_summary(
        pred_delta_y=27.6,
        std_delta_y=10.6,
        confidence="Medium",
        risk="Needs review",
        strongest_driver="Mo",
    )

    assert "+27.6 MPa" in summary
    assert "Medium" in summary
    assert "Needs review" in summary
    assert "Mo" in summary


def test_candidate_table_uses_human_readable_columns():
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
        "Candidate",
        "Expected change",
        "Uncertainty",
        "Risk-adjusted score",
        "Same-org evidence",
        "Direction support",
        "Risk flags",
        "Synthetic truth",
    ]
    assert table.iloc[0]["Expected change"] == "+12.3 MPa"
