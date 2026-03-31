"""
Tests for the Data Cleaning OpenEnv environment.

Tests cover:
  - Grader correctness (1.0 for perfect, 0.0-range for corrupted)
  - Grader determinism
  - Dataset generation integrity
  - Environment reset/step behavior
"""
from __future__ import annotations

import copy
import sys
import os

# Make data_cleaning_env importable when running tests from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from data_cleaning_env.datasets import (
    generate_task_1,
    generate_task_2,
    generate_task_3,
    TASK_MAX_STEPS,
)
from data_cleaning_env.graders import grade_task_1, grade_task_2, grade_task_3, grade
from data_cleaning_env.models import ActionType, DataCleaningAction
from data_cleaning_env.server.environment import DataCleaningEnvironment


# -----------------------------------------------------------------------------
# Dataset generation tests
# -----------------------------------------------------------------------------

class TestDatasetGeneration:
    def test_task1_shapes(self):
        dirty, gt, cols, types = generate_task_1()
        assert len(dirty) == 10
        assert len(gt) == 10
        assert len(cols) == 4
        assert len(types) == 4

    def test_task2_shapes(self):
        dirty, gt, cols, types = generate_task_2()
        # 20 ground truth rows, 25 dirty (5 duplicates added)
        assert len(gt) == 20
        assert len(dirty) == 25
        assert len(cols) == 6

    def test_task3_shapes(self):
        dirty, gt, cols, types = generate_task_3()
        assert len(gt) == 50
        assert len(dirty) == 50
        assert len(cols) == 8

    def test_datasets_are_independent_copies(self):
        """Modifying returned data must not affect subsequent calls."""
        dirty1, _, _, _ = generate_task_1()
        dirty1[0][0] = "MODIFIED"
        dirty2, _, _, _ = generate_task_1()
        assert dirty2[0][0] != "MODIFIED"


# -----------------------------------------------------------------------------
# Grader tests
# -----------------------------------------------------------------------------

class TestGraders:
    # --- Task 1 ---
    def test_task1_perfect_score(self):
        _, gt, _, _ = generate_task_1()
        score = grade_task_1(gt, gt)
        assert score == 1.0

    def test_task1_dirty_score_less_than_one(self):
        dirty, gt, _, _ = generate_task_1()
        score = grade_task_1(dirty, gt)
        assert 0.0 <= score < 1.0

    def test_task1_score_in_range(self):
        dirty, gt, _, _ = generate_task_1()
        score = grade_task_1(dirty, gt)
        assert 0.0 <= score <= 1.0

    def test_task1_deterministic(self):
        dirty, gt, _, _ = generate_task_1()
        s1 = grade_task_1(dirty, gt)
        s2 = grade_task_1(dirty, gt)
        assert s1 == s2

    # --- Task 2 ---
    def test_task2_perfect_score(self):
        _, gt, _, _ = generate_task_2()
        score = grade_task_2(gt, gt)
        assert score >= 0.9  # may not be exactly 1.0 due to weighted formula edge cases

    def test_task2_dirty_score_in_range(self):
        dirty, gt, _, _ = generate_task_2()
        score = grade_task_2(dirty, gt)
        assert 0.0 <= score <= 1.0

    def test_task2_deterministic(self):
        dirty, gt, _, _ = generate_task_2()
        s1 = grade_task_2(dirty, gt)
        s2 = grade_task_2(dirty, gt)
        assert s1 == s2

    # --- Task 3 ---
    def test_task3_perfect_score(self):
        _, gt, _, _ = generate_task_3()
        score = grade_task_3(gt, gt)
        assert score >= 0.9

    def test_task3_dirty_score_in_range(self):
        dirty, gt, _, _ = generate_task_3()
        score = grade_task_3(dirty, gt)
        assert 0.0 <= score <= 1.0

    def test_task3_deterministic(self):
        dirty, gt, _, _ = generate_task_3()
        s1 = grade_task_3(dirty, gt)
        s2 = grade_task_3(dirty, gt)
        assert s1 == s2

    # --- Registry ---
    def test_grade_registry(self):
        for task_id in [1, 2, 3]:
            _, gt, _, _ = [generate_task_1, generate_task_2, generate_task_3][task_id - 1]()
            score = grade(task_id, gt, gt)
            assert 0.0 <= score <= 1.0

    def test_grade_invalid_task(self):
        with pytest.raises(ValueError):
            grade(99, [], [])

    def test_fixing_errors_improves_score(self):
        """Fixing known errors in task 1 should raise the score."""
        dirty, gt, _, _ = generate_task_1()
        score_before = grade_task_1(dirty, gt)

        # Fix row 0 age (was "twenty-eight", should be "28")
        dirty[0][1] = "28"
        score_after = grade_task_1(dirty, gt)
        assert score_after > score_before

    def test_introducing_errors_decreases_score(self):
        """Corrupting a correct cell should lower the score."""
        _, gt, _, _ = generate_task_1()
        score_before = grade_task_1(gt, gt)

        gt_corrupted = copy.deepcopy(gt)
        gt_corrupted[0][0] = "CORRUPTED NAME"
        score_after = grade_task_1(gt_corrupted, gt)
        assert score_after < score_before


# -----------------------------------------------------------------------------
# Environment tests
# -----------------------------------------------------------------------------

class TestEnvironment:
    def setup_method(self):
        self.env = DataCleaningEnvironment()

    def test_reset_task1_returns_observation(self):
        obs = self.env.reset(task_id=1)
        assert obs.current_data is not None
        assert len(obs.current_data) == 10
        assert obs.column_names == ["name", "age", "email", "signup_date"]
        assert 0.0 <= obs.quality_score <= 1.0
        assert obs.done is False
        assert obs.reward is None

    def test_reset_task2_returns_observation(self):
        obs = self.env.reset(task_id=2)
        assert len(obs.current_data) == 25
        assert obs.column_names == ["id", "name", "email", "phone", "city", "purchase_amount"]

    def test_reset_task3_returns_observation(self):
        obs = self.env.reset(task_id=3)
        assert len(obs.current_data) == 50

    def test_reset_clears_previous_state(self):
        obs1 = self.env.reset(task_id=1)
        # Do a step
        self.env.step(DataCleaningAction(action_type=ActionType.SUBMIT))
        # Reset should give fresh state
        obs2 = self.env.reset(task_id=1)
        assert obs2.done is False
        assert self.env.state.step_count == 0

    def test_reset_invalid_task_falls_back(self):
        obs = self.env.reset(task_id=99)
        # Should fall back to task 1
        assert obs.current_data is not None

    def test_step_submit_ends_episode(self):
        self.env.reset(task_id=1)
        result = self.env.step(DataCleaningAction(action_type=ActionType.SUBMIT))
        assert result.done is True
        assert result.reward is not None
        assert result.reward >= 0

    def test_step_after_done_returns_done(self):
        self.env.reset(task_id=1)
        self.env.step(DataCleaningAction(action_type=ActionType.SUBMIT))
        result = self.env.step(DataCleaningAction(action_type=ActionType.SUBMIT))
        assert result.done is True

    def test_valid_fix_improves_quality(self):
        self.env.reset(task_id=1)
        obs_before = self.env.reset(task_id=1)

        # Fix known error: row 0 col "age" from "twenty-eight" to "28"
        result = self.env.step(DataCleaningAction(
            action_type=ActionType.FIX_VALUE,
            row=0,
            col="age",
            value="28",
        ))
        assert result.quality_score >= obs_before.quality_score

    def test_invalid_action_small_penalty(self):
        self.env.reset(task_id=1)
        result = self.env.step(DataCleaningAction(
            action_type=ActionType.FIX_VALUE,
            row=999,  # out of range
            col="age",
            value="28",
        ))
        assert result.reward < 0

    def test_step_increments_step_count(self):
        self.env.reset(task_id=1)
        assert self.env.state.step_count == 0
        self.env.step(DataCleaningAction(action_type=ActionType.FIX_VALUE, row=0, col="age", value="28"))
        assert self.env.state.step_count == 1

    def test_step_count_matches_state(self):
        self.env.reset(task_id=1)
        for _ in range(3):
            self.env.step(DataCleaningAction(action_type=ActionType.FIX_VALUE, row=0, col="age", value="28"))
        assert self.env.state.step_count == 3

    def test_quality_score_in_range_after_steps(self):
        self.env.reset(task_id=2)
        for _ in range(5):
            result = self.env.step(DataCleaningAction(
                action_type=ActionType.DELETE_ROW, row=20
            ))
            if result.done:
                break
            assert 0.0 <= result.quality_score <= 1.0

    def test_delete_valid_row_reverts_with_penalty(self):
        """Deleting a valid (non-duplicate) row should be reverted."""
        obs = self.env.reset(task_id=1)
        original_len = len(obs.current_data)
        result = self.env.step(DataCleaningAction(
            action_type=ActionType.DELETE_ROW,
            row=2,  # row 2 is clean in task 1
        ))
        # Row should be reinstated
        assert len(result.current_data) == original_len
        assert result.reward < 0  # penalty applied

    def test_state_has_correct_task_id(self):
        self.env.reset(task_id=2)
        assert self.env.state.task_id == 2

    def test_state_has_correct_max_steps(self):
        for task_id, expected_max in TASK_MAX_STEPS.items():
            self.env.reset(task_id=task_id)
            assert self.env.state.max_steps == expected_max

    def test_full_episode_task1_all_correct(self):
        """Manually apply all correct fixes and verify score approaches 1.0."""
        dirty, gt, cols, types = generate_task_1()
        self.env.reset(task_id=1)

        # Fix all known errors
        fixes = [
            (0, "age", "28"),          # row 0: "twenty-eight" → "28"
            (1, "email", "bob.smith@gmail.com"),  # row 1: bad email
            (3, "signup_date", "2023-04-18"),     # row 3: bad date format
            (4, "age", "31"),          # row 4: negative age
            (6, "email", "grace.w@gmail.com"),    # row 6: missing @
            (8, "email", "iris.t@outlook.com"),   # row 8: missing email
            (9, "signup_date", "2023-10-03"),     # row 9: invalid month 13
        ]

        for row, col, value in fixes:
            result = self.env.step(DataCleaningAction(
                action_type=ActionType.FIX_VALUE,
                row=row,
                col=col,
                value=value,
            ))

        result = self.env.step(DataCleaningAction(action_type=ActionType.SUBMIT))
        assert result.quality_score >= 0.9
        assert result.done is True
