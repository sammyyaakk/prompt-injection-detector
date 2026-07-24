# Prompt Injection Detector

Comparing a TF-IDF baseline against a sentence-embedding classifier for detecting adversarial prompts targeting LLM applications.

> Prompt injection is OWASP's top LLM security risk (LLM01). Commercial products like Lakera Guard, Rebuff, and PromptArmor address it in production. This project is a lightweight educational exploration comparing a TF-IDF baseline against a sentence-embedding classifier on ~2,000 examples from combined public datasets, with honest evaluation on a held-out set including hand-crafted adversarial examples. It is not intended to compete with production tools.

## Live demo

_Coming soon._

## Problem

_TODO_

## Approach

_TODO_

## Data

_TODO_

## Results

_TODO_

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## Setup

\`\`\`bash
git clone https://github.com/sammyyaakk/prompt-injection-detector.git
cd prompt-injection-detector
python -m venv venv
source venv/bin/activate       # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

## Usage

_TODO_

## Limitations

_TODO_

## Future work

_TODO_

## References

_TODO_