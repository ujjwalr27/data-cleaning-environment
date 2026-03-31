"""
Data Cleaning OpenEnv Environment - Models

Typed Pydantic models for Action, Observation, and State.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State


class ActionType(str, Enum):
    FIX_VALUE = "fix_value"
    FILL_MISSING = "fill_missing"
    DELETE_ROW = "delete_row"
    FIX_TYPE = "fix_type"
    SUBMIT = "submit"


class DataCleaningAction(Action):
    """
    An action the agent can take in the data cleaning environment.

    action_type: one of fix_value, fill_missing, delete_row, fix_type, submit
    row: (optional) 0-indexed row number to modify
    col: (optional) column name to modify
    value: (optional) new cell value as string
    """
    action_type: ActionType
    row: Optional[int] = None
    col: Optional[str] = None
    value: Optional[str] = None


class DataProfile(dict):
    """Stats summary for a dataset (null counts, type errors, duplicates)."""
    pass


class DataCleaningObservation(Observation):
    """
    What the agent sees after each action.

    current_data: list of rows, each a list of cell values (as strings)
    column_names: list of column header names
    column_types: expected type per column (int, float, str, date, email)
    data_profile: per-column stats (null_count, type_error_count, duplicate_rows)
    quality_score: 0.0–1.0, current data quality estimate
    message: human-readable feedback about the last action
    done: whether the episode has ended
    reward: step reward (None until first step)
    """
    current_data: List[List[str]]
    column_names: List[str]
    column_types: List[str]
    data_profile: Dict[str, Any]
    quality_score: float
    message: str
    done: bool = False
    reward: Optional[float] = None


class DataCleaningState(State):
    """
    Internal state of the episode.

    task_id: which task (1, 2, or 3)
    max_steps: maximum steps allowed
    ground_truth_hash: SHA256 of ground truth data (for integrity verification)
    """
    task_id: int = 1
    max_steps: int = 15
    ground_truth_hash: str = ""
