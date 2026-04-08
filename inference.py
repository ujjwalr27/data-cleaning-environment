"""
Data Cleaning OpenEnv Environment - Baseline Inference Script

OpenEnv Hackathon Submission - Compliant with all requirements.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

# Required environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://ujjwalml-data-cleaning-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default - optional
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional for Docker

# OpenAI Client
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai>=1.0.0")
    sys.exit(1)

# Data Cleaning env client
try:
    from data_cleaning_env import DataCleaningAction, DataCleaningEnv
except ImportError:
    print("ERROR: data_cleaning_env package not installed. Run: pip install -e .")
    sys.exit(1)



SYSTEM_PROMPT = """You are an expert data engineer. You will receive a dirty tabular dataset
and must clean it by issuing actions.

Available actions:
- fix_value: Fix a cell value (params: row, col, value)
- fill_missing: Fill an empty cell (params: row, col, value)
- delete_row: Remove a duplicate or invalid row (params: row)
- fix_type: Fix a type mismatch (params: row, col, value)
- submit: Finalize your work (no params needed)

Always respond with a valid JSON object in this exact format:
{
  "action_type": "<action>",
  "row": <int or null>,
  "col": "<column name or null>",
  "value": "<new value or null>"
}

Guidelines:
- Dates should be in YYYY-MM-DD format
- Emails must contain exactly one @ and a valid domain
- Ages/IDs/counts must be positive integers
- Salaries must be positive floats
- Status values: active, inactive, on_leave
- Departments: Engineering, Marketing, Sales, HR, Finance, Operations
- Delete only clearly duplicate rows
- Call submit when you believe the data is clean or you're running low on steps"""


def parse_llm_action(content: str) -> Optional[DataCleaningAction]:
    """Parse LLM response into a DataCleaningAction."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        data = json.loads(content)
        return DataCleaningAction(
            action_type=data.get("action_type", "submit"),
            row=data.get("row"),
            col=data.get("col"),
            value=data.get("value"),
        )
    except Exception as e:
        print(f"  [WARN] Failed to parse LLM response: {e}")
        print(f"  Raw response: {content[:200]}")
        return DataCleaningAction(action_type="submit")


def run_task(
    client: OpenAI,
    env: DataCleaningEnv,
    task_id: int,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> float:
    """Run an LLM agent on one task and return the final quality score."""
    # START log (required format)
    print("START")
    
    # Reset the environment
    step_result = env.reset(task_id=task_id)
    obs = step_result.observation
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK {task_id}")
        print(f"{'='*60}")
        print(f"Initial quality score: {obs.quality_score:.4f}")
        print(f"Dataset: {len(obs.current_data)} rows x {len(obs.column_names)} columns")
        print(f"Columns: {obs.column_names}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = obs.quality_score

    for step in range(500):  # safety cap
        if obs.done:
            break

        user_msg = f"Current dataset state (step {step}):\n\nColumns: {obs.column_names}\nTypes: {obs.column_types}\n\nData:\n"
        for i, row in enumerate(obs.current_data):
            user_msg += f"  Row {i}: {row}\n"
        user_msg += f"\nData profile: {json.dumps(obs.data_profile, indent=2)}"
        user_msg += f"\nCurrent quality score: {obs.quality_score:.4f}"
        user_msg += f"\nLast message: {obs.message}"
        user_msg += "\n\nWhat action will you take? Respond with JSON only."

        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=200,
            )
            assistant_content = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            break

        messages.append({"role": "assistant", "content": assistant_content})
        action = parse_llm_action(assistant_content)

        # STEP log (required format)
        print(f"STEP: {action.action_type}", end="")
        if action.row is not None:
            print(f" row={action.row}", end="")
        if action.col:
            print(f" col={action.col}", end="")
        if action.value:
            print(f" value='{action.value}'", end="")
        print()

        result = env.step(action)
        obs = result.observation

        if verbose:
            print(f"  Quality: {obs.quality_score:.4f} | Reward: {result.reward:.4f} | {obs.message[:80]}")

        final_score = obs.quality_score

        if obs.done:
            break

    # END log (required format)
    print("END")
    
    if verbose:
        print(f"\nTask {task_id} final score: {final_score:.4f}")
    
    return final_score


def main():
    # Use environment variables configured via OpenEnv requirements
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Get a key at https://platform.openai.com")
        sys.exit(1)
    
    # Create OpenAI client using env vars
    llm_client = OpenAI(api_key=api_key)
    
    print(f"Data Cleaning OpenEnv Baseline")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {API_BASE_URL}")
    print(f"Tasks: [1, 2, 3]")

    scores = {}
    with DataCleaningEnv(base_url=API_BASE_URL).sync() as env:
        for task_id in [1, 2, 3]:
            score = run_task(
                client=llm_client,
                env=env,
                task_id=task_id,
                model=MODEL_NAME,
                verbose=True,
            )
            scores[task_id] = score

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        difficulty = {1: "Easy", 2: "Medium", 3: "Hard"}[task_id]
        print(f"  Task {task_id} ({difficulty}): {score:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  Average:     {avg:.4f}")
    print(f"{'='*60}")

    for task_id, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Task {task_id} score {score} out of range!"
    print("\nAll scores verified in [0.0, 1.0] range.")


if __name__ == "__main__":
    main()
