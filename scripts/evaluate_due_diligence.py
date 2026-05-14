"""Evaluate public-safe AML due diligence cases.

The script validates a JSONL case file and can score model outputs when each
line contains a `model_output` field. It intentionally uses simple transparent
checks so the portfolio project can explain exactly what is being measured.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


REQUIRED_FIELDS = {
    "case_id",
    "case_type",
    "question",
    "evidence",
    "expected_risk_level",
    "required_risk_points",
    "expected_action",
    "synthetic",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            missing = REQUIRED_FIELDS - set(item)
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"{path}:{line_no}: missing fields: {missing_text}")
            cases.append(item)
    return cases


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def point_recall(output: str, required_points: list[str]) -> float:
    if not required_points:
        return 1.0
    normalized = normalize(output)
    hits = sum(1 for point in required_points if normalize(point) in normalized)
    return hits / len(required_points)


def has_grounding_signal(output: str) -> bool:
    normalized = normalize(output)
    return any(token in normalized for token in ["evidence", "source", "citation", "based on", "according to"])


def has_escalation_signal(output: str) -> bool:
    normalized = normalize(output)
    return any(token in normalized for token in ["escalate", "enhanced", "manual review", "request", "hold", "reject"])


def score_case(case: dict[str, Any]) -> dict[str, Any] | None:
    output = case.get("model_output")
    if not isinstance(output, str) or not output.strip():
        return None
    recall = point_recall(output, list(case["required_risk_points"]))
    return {
        "case_id": case["case_id"],
        "case_type": case["case_type"],
        "expected_risk_level": case["expected_risk_level"],
        "risk_point_recall": round(recall, 4),
        "has_grounding_signal": has_grounding_signal(output),
        "has_escalation_signal": has_escalation_signal(output),
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [score for case in cases if (score := score_case(case)) is not None]
    by_type: dict[str, int] = {}
    for case in cases:
        by_type[case["case_type"]] = by_type.get(case["case_type"], 0) + 1
    summary: dict[str, Any] = {
        "case_count": len(cases),
        "case_types": by_type,
        "synthetic_cases": sum(1 for case in cases if case.get("synthetic") is True),
        "scored_outputs": len(scored),
    }
    if scored:
        summary.update(
            {
                "avg_risk_point_recall": round(mean(s["risk_point_recall"] for s in scored), 4),
                "grounding_signal_rate": round(mean(1.0 if s["has_grounding_signal"] else 0.0 for s in scored), 4),
                "escalation_signal_rate": round(mean(1.0 if s["has_escalation_signal"] else 0.0 for s in scored), 4),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AML due diligence JSONL cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/eval/due_diligence_eval.jsonl"),
        help="Path to JSONL cases. Add model_output fields to score generated answers.",
    )
    parser.add_argument("--scores-out", type=Path, help="Optional path for per-case score JSONL.")
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    scores = [score for case in cases if (score := score_case(case)) is not None]
    if args.scores_out:
        args.scores_out.parent.mkdir(parents=True, exist_ok=True)
        with args.scores_out.open("w", encoding="utf-8") as handle:
            for score in scores:
                handle.write(json.dumps(score, ensure_ascii=False) + "\n")
    print(json.dumps(summarize(cases), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
