# Demo Report: AML Due Diligence Agent

This is a public-safe example of the report format this project is designed to produce. It uses synthetic evidence and should not be treated as legal, compliance, or customer-specific advice.

## Case Summary

**Entity:** Example import-export company  
**Scenario:** New customer onboarding  
**Recommended risk level:** High  
**Recommended action:** Escalate to enhanced due diligence

## Evidence Reviewed

| Evidence | Risk implication |
| --- | --- |
| Registered address changed twice in six months. | May indicate instability or attempts to avoid scrutiny. |
| Beneficial ownership documents list a nominee shareholder. | Natural-person control is not clear. |
| Payments route through three offshore intermediaries. | Transaction structure has unclear commercial rationale. |

## Risk Rationale

The case should not be cleared through standard onboarding because multiple risk indicators appear together:

- beneficial ownership is opaque;
- the entity profile changed recently;
- payment routing depends on offshore intermediaries;
- the business rationale for the payment structure is not documented.

The agent should present this as a high-risk recommendation with explicit uncertainty. The final decision remains with a human reviewer.

## Follow-Up Checks

1. Request beneficial ownership documents that identify the final natural-person controller.
2. Request trade documents, invoices, and shipping records for offshore intermediary payments.
3. Screen directors, beneficial owners, and counterparties against sanctions, PEP, and adverse-media sources.
4. Ask the relationship manager to document the commercial rationale for the payment chain.
5. Escalate to enhanced due diligence before account approval.

## Evaluation Notes

Expected model behavior:

- cite the evidence used for each risk point;
- avoid claiming that the customer is involved in money laundering;
- distinguish suspicion indicators from confirmed misconduct;
- recommend escalation instead of automatic rejection unless policy requires it;
- preserve analyst accountability for the final decision.

Useful metrics for this case:

- risk-point recall;
- citation correctness;
- hallucination rate;
- escalation correctness;
- reviewer pass/fail decision.
