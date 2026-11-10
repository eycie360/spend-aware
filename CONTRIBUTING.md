## Setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

## Check everything
ruff check . && ruff format --check . && pytest

## Workflow
- Branch from main: feature/<short-name>
- Open a PR; CI must be green before merging
