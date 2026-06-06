# Case Study: Public-Safe AML Due Diligence RAG Evaluation

This case study explains how the project can be discussed in interviews as a regulated-domain LLM application, without exposing private employer data or claiming autonomous compliance decisioning.

## Problem

AML and due diligence analysts often need to review incomplete evidence, identify risk indicators, decide whether more information is needed, and avoid overstating conclusions. A generic chatbot can produce fluent answers, but regulated workflows need grounding, uncertainty handling, escalation boundaries, and an audit trail.

This project asks a narrower question:

> Can a local RAG assistant produce analyst-useful AML review suggestions while making its evidence, risk coverage, and unsafe certainty visible?

## Public-Safe Scope

The repository uses synthetic cases only. It does not include real customer information, internal policies, private employer data, account records, sanctions-screening outputs, or confidential review decisions.

The assistant is framed as an analyst-assist workflow:

- summarize evidence;
- identify risk indicators;
- recommend follow-up checks;
- request more information when evidence is insufficient;
- route ambiguous or high-risk cases to human review.

It must not clear high-risk cases autonomously or accuse a customer of criminal conduct.

## Evaluation Design

The public evaluation set covers multiple AML and due diligence scenarios:

| Scenario | What it tests |
| --- | --- |
| Entity due diligence | Ownership opacity, offshore intermediaries, unclear commercial rationale |
| Transaction review | Structuring, threshold avoidance, rapid fund movement, virtual-asset exposure |
| Sanctions screening | Partial-name match handling and manual disambiguation |
| Adverse media | Source reliability and unverified allegations |
| PEP review | Source of wealth, source of funds, senior approval |
| Low-risk control | Avoiding unnecessary escalation when activity fits the profile |
| Trade finance | Over-invoicing, related-party trade, incomplete documents |
| Non-face-to-face onboarding | Liveness failure and device fingerprint linkage |
| Charity / NPO review | High-risk region and terrorism-financing exposure |
| Insufficient evidence | Refusal to clear based only on weak public signals |

The lightweight evaluator tracks four transparent signals when `model_output` is present:

- `risk_point_recall`: required risk indicators mentioned in the output;
- `has_grounding_signal`: output refers to evidence, sources, citations, or provided facts;
- `has_escalation_signal`: output recommends review, request, hold, rejection, or escalation where appropriate;
- `has_unsafe_certainty_signal`: output uses language that overstates legal or compliance certainty.

## Sample Evaluation Snapshot

The sample scored output file intentionally includes four acceptable outputs and one unsafe-certainty control output.

```bash
python scripts/evaluate_due_diligence.py \
  --cases examples/sample_scored_outputs.jsonl \
  --scores-out examples/sample_scores.jsonl \
  --summary-out examples/sample_summary.json
```

Current sample summary:

```json
{
  "case_count": 5,
  "case_types": {
    "entity_due_diligence": 1,
    "transaction_review": 1,
    "sanctions_screening": 1,
    "insufficient_evidence": 1,
    "unsafe_certainty_control": 1
  },
  "synthetic_cases": 5,
  "scored_outputs": 5,
  "avg_risk_point_recall": 1.0,
  "grounding_signal_rate": 1.0,
  "escalation_signal_rate": 0.8,
  "unsafe_certainty_rate": 0.2
}
```

## What This Proves

This project is useful as portfolio evidence because it shows more than a RAG demo:

- regulated-domain framing for AML and due diligence workflows;
- local-first LLM/RAG architecture for privacy-sensitive use cases;
- explicit evaluation dimensions beyond answer fluency;
- synthetic public-safe evaluation cases;
- human-in-the-loop boundaries and unsafe-certainty detection;
- reproducible scripts, tests, and CI.

## Interview Framing

A concise way to explain the project:

> I built a public-safe AML due diligence RAG prototype around local model inference, retrieval grounding, and evaluation. The main focus was not to automate compliance decisions, but to make LLM outputs auditable: whether the answer covers required risk points, refers to evidence, recommends escalation when needed, and avoids unsafe legal certainty. This connects my AML review background with LLM evaluation and regulated AI application engineering.

## Next Improvements

- Expand the synthetic evaluation set to 100+ cases.
- Add citation correctness checks instead of only grounding-signal keywords.
- Add low-risk false-escalation controls.
- Add latency and retrieval-coverage metrics.
- Add a small Streamlit evaluation dashboard for analyst review.

