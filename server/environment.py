"""
Data Cleaning OpenEnv Environment - Core Environment Logic

Implements reset(), step(), and state property per the OpenEnv spec.
"""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

from data_cleaning_env.models import (
    ActionType,
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
)
from data_cleaning_env.datasets import TASK_GENERATORS, TASK_MAX_STEPS, TASK_DESCRIPTIONS
from data_cleaning_env.graders import grade

Dataset = List[List[str]]

# Epsilon to ensure rewards are strictly within (0, 1) - must be >= 0.01 for 2-decimal formatting
REWARD_EPSILON = 0.01


def _clamp_reward(reward: float) -> float:
    """Clamp reward to be strictly between 0 and 1 (exclusive)."""
    # First clamp to [0, 1]
    reward = max(0.0, min(1.0, reward))
    # Then ensure strictly within (0, 1)
    if reward <= REWARD_EPSILON:
        return REWARD_EPSILON
    if reward >= 1.0 - REWARD_EPSILON:
        return 1.0 - REWARD_EPSILON
    return reward


def _compute_data_profile(data: Dataset, col_names: List[str], col_types: List[str]) -> Dict[str, Any]:
    """Compute per-column quality statistics."""
    profile: Dict[str, Any] = {}
    if not data or not col_names:
        return profile

    n_rows = len(data)

    for c_idx, col in enumerate(col_names):
        null_count = 0
        type_error_count = 0
        values = []

        for row in data:
            if c_idx >= len(row):
                null_count += 1
                continue
            val = row[c_idx]
            if val is None or val.strip() == "":
                null_count += 1
            else:
                values.append(val.strip())
                # Type check
                ctype = col_types[c_idx] if c_idx < len(col_types) else "str"
                if ctype == "int":
                    try:
                        int(val.strip())
                    except ValueError:
                        type_error_count += 1
                elif ctype == "float":
                    try:
                        f = float(val.strip())
                        if f < 0:
                            type_error_count += 1
                    except ValueError:
                        type_error_count += 1
                elif ctype == "email":
                    import re
                    if not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", val.strip()):
                        type_error_count += 1
                elif ctype == "date":
                    import re
                    if not re.match(r"^\d{4}-\d{2}-\d{2}$", val.strip()):
                        type_error_count += 1

        profile[col] = {
            "null_count": null_count,
            "type_error_count": type_error_count,
            "total_rows": n_rows,
        }

    # Count duplicate rows globally
    seen = set()
    dup_count = 0
    for row in data:
        key = tuple(row)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    profile["__duplicates__"] = dup_count

    return profile


def _compute_quality_score(data: Dataset, ground_truth: Dataset, task_id: int) -> float:
    """Use the task-specific grader to compute quality score."""
    return grade(task_id, data, ground_truth)


class DataCleaningEnvironment(Environment):
    """
    OpenEnv Data Cleaning Environment.

    The agent receives a dirty tabular dataset and must issue cleaning actions
    to transform it toward a clean ground-truth version.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False  # single-session for deterministic grading

    def __init__(self) -> None:
        super().__init__()
        self._state = DataCleaningState()
        self._current_data: Dataset = []
        self._ground_truth: Dataset = []
        self._col_names: List[str] = []
        self._col_types: List[str] = []
        self._task_id: int = 1
        self._done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 1,
        **kwargs,
    ) -> DataCleaningObservation:
        """
        Start a new episode.

        task_id: 1 (easy), 2 (medium), or 3 (hard). Defaults to 1.
        """
        if task_id not in TASK_GENERATORS:
            task_id = 1  # fallback to easy

        self._task_id = task_id
        dirty, ground_truth, col_names, col_types = TASK_GENERATORS[task_id]()
        self._current_data = dirty
        self._ground_truth = ground_truth
        self._col_names = col_names
        self._col_types = col_types
        self._done = False

        gt_hash = hashlib.sha256(json.dumps(ground_truth, sort_keys=True).encode()).hexdigest()

        self._state = DataCleaningState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            max_steps=TASK_MAX_STEPS[task_id],
            ground_truth_hash=gt_hash,
        )

        profile = _compute_data_profile(self._current_data, col_names, col_types)
        quality = _clamp_reward(_compute_quality_score(self._current_data, ground_truth, task_id))

        return DataCleaningObservation(
            current_data=copy.deepcopy(self._current_data),
            column_names=col_names,
            column_types=col_types,
            data_profile=profile,
            quality_score=quality,
            message=f"Task {task_id}: {TASK_DESCRIPTIONS[task_id]} Dataset ready — {len(dirty)} rows, "
                    f"{len(col_names)} columns. Starting quality score: {quality:.4f}. "
                    f"Max steps: {TASK_MAX_STEPS[task_id]}.",
            done=False,
            reward=None,
        )

    def step(
        self,
        action: DataCleaningAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> DataCleaningObservation:
        """Apply one cleaning action and return updated observation + reward."""
        if self._done:
            return DataCleaningObservation(
                current_data=copy.deepcopy(self._current_data),
                column_names=self._col_names,
                column_types=self._col_types,
                data_profile=_compute_data_profile(self._current_data, self._col_names, self._col_types),
                quality_score=_clamp_reward(_compute_quality_score(self._current_data, self._ground_truth, self._task_id)),
                message="Episode is already done. Call reset() to start a new episode.",
                done=True,
                reward=_clamp_reward(0.0),
            )

        prev_quality = _compute_quality_score(self._current_data, self._ground_truth, self._task_id)
        reward = 0.0
        message = ""

        # ---- Dispatch action ----
        act = action.action_type

        if act == ActionType.SUBMIT:
            self._done = True
            final_quality = prev_quality
            reward = final_quality * 0.5  # submit bonus
            message = (
                f"Submitted! Final quality score: {final_quality:.4f}. "
                f"Submit bonus reward: {reward:.4f}."
            )

        elif act in (ActionType.FIX_VALUE, ActionType.FILL_MISSING, ActionType.FIX_TYPE):
            row, col, value = action.row, action.col, action.value
            err = self._validate_cell_action(row, col, value)
            if err:
                reward = -0.05  # small penalty for invalid action
                message = f"Invalid action: {err}"
            else:
                c_idx = self._col_index(col)
                old_val = self._current_data[row][c_idx]
                self._current_data[row][c_idx] = value.strip()
                new_quality = _compute_quality_score(self._current_data, self._ground_truth, self._task_id)
                delta = new_quality - prev_quality

                if delta > 0:
                    reward = delta + 0.1  # correct fix bonus
                    message = f"✓ Fixed [{row}][{col}]: '{old_val}' → '{value}'. Quality improved by {delta:.4f}."
                elif delta < 0:
                    # Revert the change — penalize introduction of errors
                    self._current_data[row][c_idx] = old_val
                    reward = -0.1
                    message = f"✗ Fixing [{row}][{col}] worsened quality — action reverted."
                else:
                    message = f"~ [{row}][{col}] set to '{value}' but no quality change detected."

        elif act == ActionType.DELETE_ROW:
            row = action.row
            if row is None or row < 0 or row >= len(self._current_data):
                reward = -0.05
                message = f"Invalid row index: {row}"
            else:
                removed_row = self._current_data.pop(row)
                new_quality = _compute_quality_score(self._current_data, self._ground_truth, self._task_id)
                delta = new_quality - prev_quality

                if delta > 0:
                    reward = delta + 0.05
                    message = f"✓ Deleted row {row}. Quality improved by {delta:.4f}."
                else:
                    # Revert — put row back
                    self._current_data.insert(row, removed_row)
                    reward = -0.2  # stiff penalty for deleting valid rows
                    message = f"✗ Deleting row {row} worsened quality — reverted. Penalty applied."

        else:
            reward = -0.05
            message = f"Unknown action type: {act}"

        # ---- Increment step counter ----
        self._state.step_count += 1

        # ---- Check step limit ----
        if self._state.step_count >= self._state.max_steps and not self._done:
            self._done = True
            reward -= 0.3  # max-step penalty
            message += f" [!] Step limit ({self._state.max_steps}) reached without submitting. Penalty applied."

        # ---- Recompute observation ----
        current_quality = _clamp_reward(_compute_quality_score(self._current_data, self._ground_truth, self._task_id))
        profile = _compute_data_profile(self._current_data, self._col_names, self._col_types)

        # Clamp reward to strictly within (0, 1) as required by OpenEnv spec
        clamped_reward = _clamp_reward(reward)

        return DataCleaningObservation(
            current_data=copy.deepcopy(self._current_data),
            column_names=self._col_names,
            column_types=self._col_types,
            data_profile=profile,
            quality_score=current_quality,
            message=message,
            done=self._done,
            reward=clamped_reward,
        )

    @property
    def state(self) -> DataCleaningState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _col_index(self, col_name: str) -> int:
        """Resolve column name to index."""
        try:
            return self._col_names.index(col_name)
        except ValueError:
            raise ValueError(f"Column '{col_name}' not found. Valid columns: {self._col_names}")

    def _validate_cell_action(
        self,
        row: Optional[int],
        col: Optional[str],
        value: Optional[str],
    ) -> Optional[str]:
        """Return an error string if the action parameters are invalid, else None."""
        if row is None:
            return "row must be specified"
        if col is None:
            return "col must be specified"
        if value is None:
            return "value must be specified"
        if row < 0 or row >= len(self._current_data):
            return f"row {row} is out of range (dataset has {len(self._current_data)} rows)"
        if col not in self._col_names:
            return f"column '{col}' not found. Valid: {self._col_names}"
        return None
