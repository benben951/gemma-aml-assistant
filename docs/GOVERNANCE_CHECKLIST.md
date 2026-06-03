# Governance Checklist

This project should be presented as an analyst-assist system, not an autonomous compliance decision-maker.

The final decision remains with a human reviewer or the institution's approved compliance workflow.

## Decision Chain

A credible AML assistant should expose each step:

1. Case intake: question, entity or transaction context, known evidence, and missing evidence.
2. Retrieval: documents or snippets used for grounding.
3. Risk-point extraction: sanctions, PEP, ownership, adverse media, transaction pattern, geography, and product/channel risks.
4. Scoring or recommendation: risk level, rationale, uncertainty, and action suggestion.
5. Review action: standard monitoring, request more information, manual review, enhanced due diligence, hold, or escalation.
6. Audit record: prompt, retrieved evidence, output, score, reviewer override, and final disposition.

## Human Review Boundaries

The assistant may:

- summarize evidence
- identify risk indicators
- recommend follow-up checks
- suggest escalation when evidence is insufficient or risk is high
- explain uncertainty

The assistant must not:

- clear high-risk cases without human approval
- accuse a customer of criminal conduct
- promise regulatory compliance
- fabricate policies, data, sanctions matches, or customer facts
- turn weak evidence into legal certainty

## Evaluation Signals

The public evaluation script tracks transparent signals:

- `risk_point_recall`: whether required risk indicators appear in the output
- `has_grounding_signal`: whether the output refers to evidence, sources, or citations
- `has_escalation_signal`: whether the output recommends review, request, hold, rejection, or escalation where appropriate
- `has_unsafe_certainty_signal`: whether the output overstates legal/compliance certainty

These checks are intentionally simple and explainable. They are not a substitute for a full compliance QA program, but they make the prototype easier to audit and discuss in interviews.

## Metrics To Add Before Production

- false-clear rate for high-risk cases
- false-escalation rate for low-risk controls
- citation correctness
- missing-evidence refusal quality
- analyst override rate
- review-time savings
- latency and retrieval coverage
- drift across policy updates and new typologies

## Data And Privacy Rules

For a public portfolio version:

- use synthetic cases only
- avoid real customers, counterparties, account numbers, documents, and internal policies
- do not commit secrets or model provider keys
- keep private evidence stores outside git
- clearly mark demo outputs as non-legal and non-regulatory advice

For a private enterprise version:

- log only what is needed for audit
- define retention rules
- separate reviewer identity from public artifacts
- redact personally identifiable information in exported reports
- require manual approval for destructive or customer-impacting actions

## Resume-Safe Framing

Use this wording:

> Built a public-safe AML due diligence RAG prototype with local model inference, retrieval grounding, citation-aware responses, and an evaluation layer for risk-point recall, grounding, escalation behavior, and unsafe-certainty detection.

Avoid this wording:

> Built an AI system that automatically detects money laundering and decides whether customers are safe.
