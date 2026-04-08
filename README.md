---
title: Data Cleaning OpenEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
pinned: false
---

# Data Cleaning OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces)

A real-world **OpenEnv** environment where an AI agent must clean dirty tabular datasets. Modeled after the data cleaning task that occupies ~80% of a data engineer's time, this environment provides:

- **3 graded tasks** from easy type fixes to hard cross-column reconciliation
- **Partial-progress rewards** at every step (not sparse binary)
- **Deterministic graders** with clear success criteria
- **Reproducible baseline** via OpenAI API

---

## 🌍 Environment Description

The agent receives a dirty CSV-like dataset and must issue targeted **cleaning actions** to transform it into a clean ground-truth version. After each action, the environment scores data quality and returns feedback.

**Why data cleaning?**
- Every ML practitioner does this daily — genuine, unsolved real-world task
- Natural partial-progress structure (fix one error = small reward)
- Clear deterministic grading (compare to known ground truth)
- Novel domain not yet in OpenEnv hub

---

## 📐 Action Space

The agent can issue one of 5 action types per step:

| Action | Description | Required Params |
|--------|-------------|-----------------|
| `fix_value` | Replace a specific cell value | `row`, `col`, `value` |
| `fill_missing` | Fill an empty/null cell | `row`, `col`, `value` |
| `delete_row` | Remove a duplicate/invalid row | `row` |
| `fix_type` | Cast cell to correct type | `row`, `col`, `value` |
| `submit` | Finalize — triggers final grading | _(none)_ |

All params use 0-indexed rows and string column names.

---

## 👁️ Observation Space

After each action, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `current_data` | `list[list[str]]` | Current dataset (rows × columns) |
| `column_names` | `list[str]` | Column header names |
| `column_types` | `list[str]` | Expected type per column |
| `data_profile` | `dict` | Per-column stats (null_count, type_errors, duplicates) |
| `quality_score` | `float` | Current quality 0.0–1.0 |
| `message` | `str` | Feedback on last action |
| `done` | `bool` | Whether episode ended |
| `reward` | `float` | Step reward |

---

## 📋 Tasks

### Task 1 — Easy: "Fix the Basics"
- **Rows/Cols**: 10 rows × 4 cols (name, age, email, signup_date)
- **Errors**: ~5 obvious errors — wrong age type (`"twenty-eight"`), bad email format,   invalid date month/format, negative age
- **Grader**: Fraction of cells matching ground truth
- **Max steps**: 15

### Task 2 — Medium: "Clean the Customer List"
- **Rows/Cols**: 25 rows × 6 cols (id, name, email, phone, city, purchase_amount)
- **Errors**: ~12 errors — missing email/city/amount, exact duplicate rows, bad phone format
- **Grader**: Weighted — 40% cell accuracy + 30% deduplication + 30% missing value handling
- **Max steps**: 30

### Task 3 — Hard: "Enterprise Data Reconciliation"
- **Rows/Cols**: 50 rows × 8 cols (employee_id, name, department, salary, manager_id, start_date, email, status)
- **Errors**: ~25 errors — referential integrity (manager_id → non-existent employee), logical violations (negative salary, future dates), invalid statuses/departments, malformed emails, missing names
- **Grader**: Weighted — 30% cell accuracy + 25% referential integrity + 25% logical consistency + 20% deduplication
- **Max steps**: 60

---

## 🏆 Reward Function

| Event | Reward |
|-------|--------|
| Correct fix (quality improves) | `Δ quality + 0.1` bonus |
| Neutral fix (no quality change) | `0.0` |
| Introducing error (quality drops) | `-0.1` (action reverted) |
| Deleting valid row | `-0.2` (action reverted) |
| Invalid action params | `-0.05` |
| Submit (episode end) | `+final_quality × 0.5` |
| Step limit exceeded | `-0.3` |

---

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)
- `openenv-core` package

### Install

```bash
pip install openenv-core
pip install git+https://huggingface.co/spaces/<your-username>/data-cleaning-env
```

### Run Locally (without Docker)

```bash
pip install -e .
uvicorn data_cleaning_env.server.app:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Interact

```python
from data_cleaning_env import DataCleaningAction, DataCleaningEnv

with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
    # Start Task 1 (easy)
    obs = env.reset(task_id=1)
    print(f"Quality: {obs.quality_score}")
    print(f"Data:\n{obs.current_data}")

    # Fix a cell
    result = env.step(DataCleaningAction(
        action_type="fix_value",
        row=0,
        col="age",
        value="28"
    ))
    print(f"New quality: {result.observation.quality_score}")

    # Submit when done
    result = env.step(DataCleaningAction(action_type="submit"))
    print(f"Final score: {result.observation.quality_score}")
```

### Run Baseline

```bash
HF_TOKEN=<your-hf-token> python inference.py
```

---

## 📊 Baseline Scores

Scores produced by `gpt-4o-mini` with temperature=0 (deterministic):

| Task | Difficulty | Baseline Score |
|------|-----------|----------------|
| 1    | Easy      | ~0.85          |
| 2    | Medium    | ~0.65          |
| 3    | Hard      | ~0.40          |

*(Scores will be updated after running the baseline script)*

---

## 📁 Project Structure

```
data_cleaning_env/
├── __init__.py           # Public API exports
├── models.py             # Pydantic Action, Observation, State
├── client.py             # EnvClient (typed HTTP client)
├── datasets.py           # Deterministic dirty+ground-truth datasets
├── graders.py            # Task-specific grader functions
├── inference.py          # OpenAI API baseline inference script
├── openenv.yaml          # Environment manifest
├── pyproject.toml        # Package metadata & dependencies
├── Dockerfile            # Container image for HF Spaces
├── README.md             # This file
├── server/
│   ├── __init__.py
│   ├── environment.py    # Core environment logic (reset/step/state)
│   ├── app.py            # FastAPI application
│   └── requirements.txt  # Docker dependencies
└── tests/
    └── test_environment.py
```

---

## 🔬 Validation

```bash
# Validate OpenEnv spec compliance
openenv validate

# Docker build & health check
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env &
curl http://localhost:7860/health

# Run tests
pytest tests/ -v
```

---

## 📝 License

MIT
