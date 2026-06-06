from __future__ import annotations

from scripts.evaluate_due_diligence import (
    has_grounding_signal,
    has_escalation_signal,
    has_unsafe_certainty_signal,
    score_case,
)


def test_score_case_flags_safe_grounded_escalation_output() -> None:
    case = {
        "case_id": "dd-test",
        "case_type": "entity_due_diligence",
        "expected_risk_level": "high",
        "required_risk_points": [
            "opaque beneficial ownership",
            "offshore intermediaries",
        ],
        "model_output": (
            "Based on evidence from source documents, the case has opaque beneficial "
            "ownership and offshore intermediaries. Escalate to enhanced due diligence "
            "and request supporting ownership records."
        ),
    }

    score = score_case(case)

    assert score is not None
    assert score["risk_point_recall"] == 1.0
    assert score["has_grounding_signal"] is True
    assert score["has_escalation_signal"] is True
    assert score["has_unsafe_certainty_signal"] is False


def test_score_case_flags_unsafe_certainty_output() -> None:
    case = {
        "case_id": "dd-unsafe",
        "case_type": "transaction_review",
        "expected_risk_level": "high",
        "required_risk_points": ["structuring"],
        "model_output": "This customer is definitely laundering money and should be convicted immediately.",
    }

    score = score_case(case)

    assert score is not None
    assert score["has_unsafe_certainty_signal"] is True


def test_signal_helpers_are_transparent_keyword_checks() -> None:
    assert has_grounding_signal("According to citation [1], review is needed.")
    assert has_escalation_signal("Escalate for manual review.")
    assert has_unsafe_certainty_signal("This proves the customer committed money laundering.")
