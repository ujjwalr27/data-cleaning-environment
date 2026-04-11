"""
Data Cleaning OpenEnv Environment - Reward Computation

Dedicated reward module following the FinQA pattern from OpenEnv reference
environments. Separates reward logic from environment state management for
clarity, testability, and demonstrating intentional reward design.

Reward philosophy:
  - Partial progress at every step (not binary end-of-episode)
  - Positive reinforcement for correct fixes
  - Penalties that discourage destructive/invalid actions
  - Submit bonus with efficiency scaling
"""
from __future__ import annotations

# Epsilon to ensure rewards are strictly within (0, 1)
REWARD_EPSILON = 0.01


def clamp_reward(reward: float) -> float:
    """Clamp reward to be strictly between 0 and 1 (exclusive).

    The OpenEnv validator rejects reward values of exactly 0.0 or 1.0,
    so we ensure all returned rewards are in the open interval (0, 1).
    """
    reward = max(0.0, min(1.0, reward))
    if reward <= REWARD_EPSILON:
        return REWARD_EPSILON
    if reward >= 1.0 - REWARD_EPSILON:
        return 1.0 - REWARD_EPSILON
    return reward


def compute_fix_reward(
    delta_quality: float,
    is_valid_action: bool,
) -> float:
    """Compute reward for a fix_value / fill_missing / fix_type action.

    Args:
        delta_quality: Change in quality score (new - old). Positive = improvement.
        is_valid_action: Whether the action params were valid (row/col/value exist).

    Returns:
        Raw reward (will be clamped later by the environment).
    """
    if not is_valid_action:
        return -0.05  # small penalty for invalid params

    if delta_quality > 0:
        return delta_quality + 0.1  # improvement bonus
    elif delta_quality < 0:
        return -0.1  # worsened data — action will be reverted
    else:
        return 0.0  # no change


def compute_delete_reward(delta_quality: float) -> float:
    """Compute reward for a delete_row action.

    Args:
        delta_quality: Change in quality score after row deletion.

    Returns:
        Raw reward. Negative if the deletion hurt quality.
    """
    if delta_quality > 0:
        return delta_quality + 0.05  # smaller bonus than cell fixes
    else:
        return -0.2  # stiff penalty for deleting valid rows


def compute_submit_reward(
    final_quality: float,
    steps_used: int,
    max_steps: int,
) -> float:
    """Compute the submit bonus.

    Rewards both final data quality and efficiency (fewer steps = better).

    Args:
        final_quality: Quality score at time of submission (0-1).
        steps_used: Number of steps the agent used before submitting.
        max_steps: Maximum allowed steps for this task.

    Returns:
        Submit reward = quality_bonus + efficiency_bonus.
    """
    steps_used_ratio = steps_used / max(max_steps, 1)
    efficiency_bonus = 0.1 * max(0.0, 1.0 - steps_used_ratio)
    return final_quality * 0.5 + efficiency_bonus


def compute_step_limit_penalty() -> float:
    """Penalty when the agent exhausts all steps without submitting."""
    return -0.3


def compute_invalid_action_penalty() -> float:
    """Penalty for unknown/unsupported action types."""
    return -0.05
