"""
Data Cleaning OpenEnv Environment

A real-world OpenEnv environment for training and evaluating AI agents on
tabular data cleaning tasks ranging from basic type fixes to complex
cross-column reconciliation.

Quick start:
    >>> from data_cleaning_env import DataCleaningAction, DataCleaningEnv
    >>> with DataCleaningEnv(base_url="http://localhost:7860") as env:
    ...     result = env.reset(task_id=1)
    ...     result = env.step(DataCleaningAction(action_type="submit"))
"""
from typing import Any

from data_cleaning_env.models import (
    ActionType,
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
)

__all__ = [
    "ActionType",
    "DataCleaningAction",
    "DataCleaningObservation",
    "DataCleaningState",
    "DataCleaningEnv",
]


def __getattr__(name: str) -> Any:
    """Lazy import for the client to avoid heavy httpx import at package load."""
    if name == "DataCleaningEnv":
        from data_cleaning_env.client import DataCleaningEnv as _DataCleaningEnv
        return _DataCleaningEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
