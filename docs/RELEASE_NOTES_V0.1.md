# v0.1 Release Notes

Gemma AML Compliance Assistant v0.1 packages a public-safe AML and due-diligence RAG evaluation prototype.

## Highlights

- Public-safe AML and due-diligence evaluation cases.
- Scoring for risk-point coverage, grounding signals, escalation signals, and unsafe certainty.
- Local-first RAG project framing with Gemma-style inference and Qdrant-backed retrieval.
- Streamlit app scaffold and governance notes.
- Sample scored outputs and aggregate summary artifacts.
- CI-backed tests.

## Good First Review Path

1. Run the evaluation quickstart from `README.md`.
2. Read `docs/DEMO_REPORT.md`.
3. Inspect `examples/sample_summary.json`.
4. Run `python -m pytest -q`.

## Boundary

The repo uses public-safe synthetic examples and does not include real customer, employer, or regulated case data.
