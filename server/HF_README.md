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

A real-world **OpenEnv** environment where an AI agent cleans dirty tabular datasets.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| 1 | Easy | Fix type/format errors (10 rows) |
| 2 | Medium | Handle missing values + deduplication (25 rows) |
| 3 | Hard | Cross-column referential integrity (50 rows) |

## Quick Start

```python
pip install openenv-core
```

```python
from openenv.core.env_client import EnvClient

with EnvClient(base_url="https://ujjwalml-data-cleaning-env.hf.space").sync() as env:
    result = env.reset(task_id=1)
    print(f"Quality: {result.observation.quality_score}")
```

## Endpoints

- `POST /reset` — start a new episode
- `POST /step` — apply a cleaning action
- `GET /health` — health check
