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

# 🧹 Data Cleaning OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)](https://huggingface.co/spaces)

A real-world **OpenEnv** environment where an AI agent must clean dirty tabular datasets. Modeled after the data cleaning task that occupies ~80% of a data engineer's time — the most common yet tedious real-world data task.

## 🎯 Why This Environment?

| Criteria | How We Address It |
|---|---|
| **Real-world utility (30%)** | Data cleaning is the #1 bottleneck in ML pipelines — every practitioner does this daily |
| **Task & grader quality (25%)** | 3 tasks (easy → hard) with weighted, multi-dimensional graders + fuzzy partial credit |
| **Environment design (20%)** | Clean state management, partial-progress rewards, typed action/observation spaces |
| **Code quality (15%)** | Full OpenEnv spec, typed Pydantic models, separated rewards module, proper client |
| **Creativity (10%)** | Novel domain not yet in OpenEnv hub, type-aware fuzzy matching for smoother RL signal |

---

## ⚡ Quick Start

### Install

```bash
pip install openenv-core
pip install git+https://huggingface.co/spaces/ujjwalml/data-cleaning-env
```

### Use the Client

```python
from data_cleaning_env import DataCleaningAction, DataCleaningEnv

# Connect to the running environment
with DataCleaningEnv(base_url="https://ujjwalml-data-cleaning-env.hf.space") as env:
    # Start Task 1 (easy)
    result = env.reset(task_id=1)
    print(f"Quality: {result.observation.quality_score}")
    print(f"Columns: {result.observation.column_names}")

    # Fix a cell
    result = env.step(DataCleaningAction(
        action_type="fix_value",
        row=2, col="age", value="28"
    ))
    print(f"Reward: {result.reward}")
    print(f"Message: {result.observation.message}")

    # Submit when done
    result = env.step(DataCleaningAction(action_type="submit"))
    print(f"Final quality: {result.observation.quality_score}")
```

### Run Locally

```bash
# Clone and install
git clone https://huggingface.co/spaces/ujjwalml/data-cleaning-env
cd data-cleaning-env
pip install -e .

# Start the server
uvicorn data_cleaning_env.server.app:app --host 0.0.0.0 --port 7860

# Or use Docker
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Run Baseline Inference

```bash
export HF_TOKEN=<your-hf-token>
python inference.py
```

---

## 📐 Action Space

The agent can issue one of 5 discrete action types per step:

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
| `column_types` | `list[str]` | Expected type per column (int, float, str, date, email) |
| `data_profile` | `dict` | Per-column stats: null_count, type_error_count, duplicates |
| `quality_score` | `float` | Current data quality (0.0–1.0) |
| `message` | `str` | Feedback on last action |
| `done` | `bool` | Whether episode ended |
| `reward` | `float` | Step reward |

---

## 📋 Tasks (Easy → Medium → Hard)

### Task 1 — "Fix the Basics" (Easy)
- **Dataset**: 10 rows × 4 cols (`name`, `age`, `email`, `signup_date`)
- **Errors**: ~5 obvious — wrong age type (`"twenty-eight"`), bad email, invalid date
- **Grader**: 70% exact cell accuracy + 30% fuzzy matching
- **Max steps**: 15

### Task 2 — "Clean the Customer List" (Medium)
- **Dataset**: 25 rows × 6 cols (`id`, `name`, `email`, `phone`, `city`, `purchase_amount`)
- **Errors**: ~12 — missing values, 5 duplicate rows, bad phone formats
- **Grader**: 35% cell accuracy + 15% fuzzy + 30% deduplication + 20% missing value fill
- **Max steps**: 30

### Task 3 — "Enterprise Data Reconciliation" (Hard)
- **Dataset**: 50 rows × 8 cols (`employee_id`, `name`, `department`, `salary`, `manager_id`, `start_date`, `email`, `status`)
- **Errors**: ~25 — referential integrity violations, negative salaries, future dates, invalid departments/statuses, malformed emails
- **Grader**: 25% cell accuracy + 10% fuzzy + 25% referential integrity + 25% logical consistency + 15% deduplication
- **Max steps**: 60

---

## 🏆 Reward Function

Rewards provide **partial progress signal at every step** (not binary end-of-episode):

| Event | Reward | Rationale |
|-------|--------|-----------|
| Correct fix (quality ↑) | `Δ quality + 0.1` | Encourage fixing errors |
| Neutral fix (quality =) | `0.0` | No penalty for exploration |
| Introducing error (quality ↓) | `-0.1` (reverted) | Discourage destructive actions |
| Deleting valid row | `-0.2` (reverted) | Stiff penalty for data loss |
| Invalid action params | `-0.05` | Minor penalty for bad input |
| Submit (episode end) | `quality × 0.5 + efficiency` | Reward both quality and speed |
| Efficiency bonus | `0.1 × (1 - steps/max)` | Fewer steps = more bonus |
| Step limit exceeded | `-0.3` | Must learn to submit in time |

> Reward computation is implemented in a dedicated `server/rewards.py` module
> following the FinQA pattern from the OpenEnv reference repository.

---

## 📊 Baseline Scores

**Initial dirty quality** (before any agent action):

| Task | Difficulty | Initial Quality | Max Steps |
|------|-----------|-----------------|-----------|
| 1 | Easy | ~0.83 | 15 |
| 2 | Medium | ~0.38 | 30 |
| 3 | Hard | ~0.95 | 60 |

**Estimated `gpt-4o-mini` baseline** (temperature=0):

| Task | Baseline Quality | Steps Used | Success |
|------|-----------------|------------|---------|
| 1 | ~0.92 | 6–8 | ✅ |
| 2 | ~0.72 | 15–20 | ✅ |
| 3 | ~0.96 | 20–30 | ✅ |

---

## 📁 Project Structure

```
data_cleaning_env/
├── __init__.py           # Public API exports (lazy client import)
├── models.py             # Pydantic Action, Observation, State types
├── client.py             # HTTP client (context manager, typed methods)
├── datasets.py           # Deterministic dirty + ground-truth dataset generators
├── graders.py            # Task-specific graders with fuzzy matching
├── inference.py          # OpenAI API baseline inference script
├── openenv.yaml          # Environment manifest with task metadata
├── pyproject.toml        # Package metadata & dependencies
├── Dockerfile            # Container image for HF Spaces
├── README.md             # This file
├── server/
│   ├── __init__.py
│   ├── environment.py    # Core environment logic (reset/step/state)
│   ├── rewards.py        # Dedicated reward computation module
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

# Run unit tests
pytest tests/ -v
```

---

## 📝 License

MIT
