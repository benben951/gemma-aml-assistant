# Portfolio Roadmap

This roadmap turns the repository from a RAG demo into a credible LLM evaluation project for AML and due diligence workflows.

## V0: Public-Safe Evaluation Layer

Status: in progress

- Add synthetic due diligence cases.
- Add a transparent scoring script.
- Add a sample analyst-style report.
- Document the bad-case taxonomy and target metrics.

## V1: Analyst Workflow Demo

Target artifacts:

- Streamlit demo screenshots.
- A short demo video or GIF.
- One English risk report and one Chinese risk report.
- Case-level logs showing retrieval evidence, model output, risk score, and reviewer action.

## V2: Evaluation Dashboard

Target artifacts:

- Risk-point recall by case type.
- Citation accuracy review table.
- Hallucination and overconfidence failure counts.
- Latency and cost estimate per case.
- Human reviewer pass/fail column.

## V3: Agentic Due Diligence Workflow

Target artifacts:

- Extractor agent for entity and transaction facts.
- Investigator agent for evidence retrieval.
- Scorer agent for risk indicators.
- Verifier agent for citation and uncertainty checks.
- Reporter agent for analyst-facing summary.

## V4: Post-Training Bridge

Target artifacts:

- Convert reviewed cases into instruction-tuning examples.
- Build preference pairs from good and bad outputs.
- Compare baseline prompt, RAG, and fine-tuned small-model outputs.
- Track quality metrics before and after tuning.

## Safety Boundaries

- Use only synthetic or public-safe cases.
- Do not include real customer data, internal policies, credentials, or private company materials.
- Keep the system assistant-only: it recommends and escalates, while a human reviewer makes final decisions.
