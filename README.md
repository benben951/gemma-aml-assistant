# Gemma AML Compliance Assistant

[![CI](https://github.com/benben951/gemma-aml-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/benben951/gemma-aml-assistant/actions/workflows/ci.yml)

Offline RAG and evaluation project for AML and due diligence knowledge workflows.

## Portfolio Snapshot

This repository is positioned as an AML and due diligence RAG evaluation project for regulated financial workflows. It demonstrates local LLM deployment, retrieval grounding, citation-aware responses, and an evaluation layer for hallucination and risk-coverage review.

- Portfolio angle: AI application engineering for AML, KYC, and due diligence workflows
- Evaluation focus: grounding, citation accuracy, risk-point coverage, uncertainty handling, and analyst actionability
- Main portfolio hub: [AI Trust & Agent Evaluation Portfolio](https://github.com/benben951/ai-trust-agent-evaluation-portfolio)
- Supporting docs: [docs/CASE_STUDY.md](docs/CASE_STUDY.md), [docs/EVALUATION.md](docs/EVALUATION.md), [docs/GOVERNANCE_CHECKLIST.md](docs/GOVERNANCE_CHECKLIST.md), [docs/DEMO_REPORT.md](docs/DEMO_REPORT.md), [docs/PROJECT_ROADMAP.md](docs/PROJECT_ROADMAP.md)

## Why This Project Exists

Compliance assistants are not just "chat with documents" systems. In regulated workflows they need to be:

- local-first when privacy matters
- grounded in retrieved evidence
- explicit about uncertainty
- inspectable enough for analyst review

This project explores that design space with Gemma 4 as the local reasoning model.

## What It Does

- Runs a local RAG workflow for AML and due diligence questions
- Uses retrieval plus citation-aware generation instead of free-form answer synthesis
- Adds explainability signals such as confidence and supporting evidence
- Includes evaluation cases for answer grounding, coverage, escalation behavior, and unsafe certainty checks

## System Components

- local LLM inference with Gemma 4
- vector retrieval with Qdrant
- QA orchestration and response formatting
- evaluation scripts and public-safe demo artifacts
- Streamlit interface for a simple analyst-facing prototype

## Quick Start

```bash
git clone https://github.com/benben951/gemma-aml-assistant.git
cd gemma-aml-assistant
docker compose up -d
```

For local model serving:

```bash
ollama pull gemma4:26b-a4b
```

Then run the app or evaluation scripts from the repo.

## Verification

```bash
python -m pytest -q
python scripts/evaluate_due_diligence.py --cases data/eval/due_diligence_eval.jsonl
python scripts/evaluate_due_diligence.py \
  --cases examples/sample_scored_outputs.jsonl \
  --scores-out examples/sample_scores.jsonl \
  --summary-out examples/sample_summary.json
```

## Sample Evaluation Snapshot

The sample scored-output file includes four acceptable synthetic outputs and one unsafe-certainty control output.

```json
{
  "case_count": 5,
  "synthetic_cases": 5,
  "scored_outputs": 5,
  "avg_risk_point_recall": 1.0,
  "grounding_signal_rate": 1.0,
  "escalation_signal_rate": 0.8,
  "unsafe_certainty_rate": 0.2
}
```

## Project Structure

```text
src/              data models, retrieval logic, Gemma client, QA pipeline
app/              Streamlit app
data/eval/        AML and due-diligence evaluation cases
tests/            unit and integration-oriented checks
docs/             evaluation notes, demo report, roadmap
```

## Resume Angle

Built an offline AML and due diligence assistant with local Gemma inference, Qdrant-backed retrieval, citation-aware responses, and evaluation cases for grounding, risk-point coverage, escalation behavior, and unsafe-certainty detection in regulated workflows.
