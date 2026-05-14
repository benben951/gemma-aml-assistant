# Evaluation Notes

This project is designed as a public-safe prototype for AML and due diligence question answering. The goal is not only to generate fluent answers, but to evaluate whether answers are grounded, traceable, and useful for regulated financial workflows.

## What The Eval Checks

The evaluation set should cover four dimensions:

| Dimension | What it means | Example signal |
| --- | --- | --- |
| Grounding | The answer uses retrieved evidence instead of unsupported claims. | Citations match the generated statement. |
| Risk coverage | Key AML/DD risk points are mentioned. | Sanctions, ownership, transaction anomalies, geography, adverse media. |
| Refusal and uncertainty | The assistant avoids pretending when evidence is missing. | Explicit uncertainty or request for more evidence. |
| Actionability | The answer helps an analyst decide the next review step. | Risk level, rationale, and follow-up checks. |

## Current Public Eval Shape

- Domain: AML / KYC / due diligence knowledge QA.
- Format: JSONL cases with question, expected risk points, and grounding notes.
- Current role: smoke and demonstration set.
- Next target: expand to 100-200 synthetic but realistic cases across entity due diligence, transaction review, sanctions screening, adverse media, and beneficial ownership.

Public files:

- `data/eval/aml_eval.jsonl`: knowledge-oriented AML QA cases.
- `data/eval/due_diligence_eval.jsonl`: synthetic case-review examples for risk-point recall and escalation behavior.
- `scripts/evaluate_due_diligence.py`: transparent JSONL validator and lightweight scorer.

Quick check:

```bash
python scripts/evaluate_due_diligence.py --cases data/eval/due_diligence_eval.jsonl
```

## Bad-Case Taxonomy

When reviewing outputs, classify failures into:

- `missing_evidence`: answer has claims but no useful citation.
- `wrong_evidence`: citation does not support the claim.
- `risk_omission`: answer misses an important AML/DD risk point.
- `overconfident`: answer sounds certain when retrieved evidence is weak.
- `generic_answer`: answer is correct but not useful for analyst workflow.
- `unsafe_advice`: answer gives operational or legal certainty beyond the available evidence.

## Target Metrics

For a public portfolio version, track:

- citation accuracy;
- expected risk-point recall;
- hallucination rate;
- refusal quality for insufficient evidence;
- human review pass rate;
- latency per question.

## Why This Matters

Financial risk workflows need traceability more than generic chat quality. A usable AML assistant should expose where the answer came from, why a risk level was assigned, and what an analyst should verify next. This evaluation layer is the bridge from a demo RAG app to an enterprise-grade LLM application.
