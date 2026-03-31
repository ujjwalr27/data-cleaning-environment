"""
Data Cleaning OpenEnv Environment - Client

Typed client for interacting with the Data Cleaning environment.
"""
from __future__ import annotations

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from data_cleaning_env.models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
)


class DataCleaningEnv(EnvClient[DataCleaningAction, DataCleaningObservation, DataCleaningState]):
    """
    Client for the Data Cleaning OpenEnv environment.

    Usage (sync):
        with DataCleaningEnv(base_url="https://...").sync() as env:
            obs = env.reset(task_id=1)
            result = env.step(DataCleaningAction(
                action_type="fix_value", row=0, col="age", value="28"
            ))

    Usage (async):
        async with DataCleaningEnv(base_url="https://...") as env:
            obs = await env.reset(task_id=1)
            result = await env.step(DataCleaningAction(
                action_type="submit"
            ))
    """

    def _step_payload(self, action: DataCleaningAction) -> dict:
        """Serialize action to wire format."""
        payload = {"action_type": action.action_type.value}
        if action.row is not None:
            payload["row"] = action.row
        if action.col is not None:
            payload["col"] = action.col
        if action.value is not None:
            payload["value"] = action.value
        return payload

    def _parse_result(self, payload: dict) -> StepResult:
        """Deserialize server response to StepResult."""
        obs_data = payload.get("observation", {})
        obs = DataCleaningObservation(
            current_data=obs_data.get("current_data", []),
            column_names=obs_data.get("column_names", []),
            column_types=obs_data.get("column_types", []),
            data_profile=obs_data.get("data_profile", {}),
            quality_score=obs_data.get("quality_score", 0.0),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DataCleaningState:
        """Deserialize state from server response."""
        return DataCleaningState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", 1),
            max_steps=payload.get("max_steps", 15),
            ground_truth_hash=payload.get("ground_truth_hash", ""),
        )
