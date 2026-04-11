"""
Data Cleaning OpenEnv Environment - HTTP Client

Typed client for interacting with the Data Cleaning environment server.
Follows the reference patterns from calendar_env, coding_env, and tbench2_env.

Usage (sync — context manager):
    with DataCleaningEnv(base_url="https://your-space.hf.space") as env:
        result = env.reset(task_id=1)
        print(result.observation.quality_score)

        result = env.step(DataCleaningAction(
            action_type="fix_value", row=0, col="age", value="28"
        ))
        print(result.observation.message)

        result = env.step(DataCleaningAction(action_type="submit"))
        print(f"Final quality: {result.observation.quality_score}")

Usage (direct):
    env = DataCleaningEnv(base_url="http://localhost:7860")
    result = env.reset(task_id=1)
    result = env.step(...)
    state = env.state()
    env.close()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

import httpx

from data_cleaning_env.models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
)

ObsT = TypeVar("ObsT")


@dataclass
class StepResult(Generic[ObsT]):
    """Result from a step or reset call."""
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False


class DataCleaningEnv:
    """HTTP client for the Data Cleaning OpenEnv environment.

    Mirrors the reference client pattern used by calendar_env, coding_env,
    reasoning_gym_env, and tbench2_env in the official OpenEnv repository.

    Args:
        base_url: URL of the running environment server (e.g. HF Space URL).
        timeout_s: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client: Optional[httpx.Client] = None

    # ---- Context manager ----

    def __enter__(self) -> "DataCleaningEnv":
        self._ensure_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release HTTP resources."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    # ---- OpenEnv API ----

    def reset(
        self,
        task_id: int = 1,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> StepResult[DataCleaningObservation]:
        """Reset the environment and start a new episode.

        Args:
            task_id: Task difficulty — 1 (easy), 2 (medium), or 3 (hard).
            seed: Optional random seed for reproducibility.
            episode_id: Optional custom episode ID.

        Returns:
            StepResult containing the initial observation.
        """
        payload: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id

        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/reset",
            json=payload,
        )
        response.raise_for_status()
        return self._parse_step_result(response.json())

    def step(
        self,
        action: DataCleaningAction,
    ) -> StepResult[DataCleaningObservation]:
        """Execute one cleaning action in the environment.

        Args:
            action: The action to take (fix_value, fill_missing, delete_row,
                    fix_type, or submit).

        Returns:
            StepResult with updated observation, reward, and done flag.
        """
        payload = self._step_payload(action)
        client = self._ensure_client()
        response = client.post(
            f"{self.base_url}/step",
            json=payload,
        )
        response.raise_for_status()
        return self._parse_step_result(response.json())

    def state(self) -> DataCleaningState:
        """Get the current episode state (step count, task, etc.).

        Returns:
            DataCleaningState with episode metadata.
        """
        client = self._ensure_client()
        response = client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return self._parse_state(response.json())

    def health(self) -> bool:
        """Check if the environment server is healthy.

        Returns:
            True if the server responds with 200.
        """
        try:
            client = self._ensure_client()
            response = client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    # ---- Serialization helpers ----

    def _step_payload(self, action: DataCleaningAction) -> dict:
        """Serialize action to wire format."""
        payload: Dict[str, Any] = {"action_type": action.action_type.value}
        if action.row is not None:
            payload["row"] = action.row
        if action.col is not None:
            payload["col"] = action.col
        if action.value is not None:
            payload["value"] = action.value
        return payload

    def _parse_step_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[DataCleaningObservation]:
        """Deserialize server response to StepResult."""
        obs_data = payload.get("observation", {})
        observation = DataCleaningObservation(
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
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DataCleaningState:
        """Deserialize state from server response."""
        return DataCleaningState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", 1),
            max_steps=payload.get("max_steps", 15),
            ground_truth_hash=payload.get("ground_truth_hash", ""),
        )
