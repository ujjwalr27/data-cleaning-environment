"""
Data Cleaning OpenEnv Environment - Baseline Inference Script

Runs an LLM agent against all 3 tasks and reports reproducible baseline scores.

Usage (OpenAI - paid):
    OPENAI_API_KEY=<key> python inference.py --base-url <url>

Usage (Groq - FREE at console.groq.com):
    GROQ_API_KEY=<key> python inference.py --provider groq --base-url <url>

Default base_url: http://localhost:7860
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

# OpenAI Client (also used for Groq which is OpenAI-compatible)
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
    print(f"\n{'='*60}")
    print(f"TASK {task_id}")
    print(f"{'='*60}")

    # Reset the environment
    step_result = env.reset(task_id=task_id)
    obs = step_result.observation
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

        if verbose:
            print(f"\nStep {step + 1}: {action.action_type}", end="")
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

    print(f"\nTask {task_id} final score: {final_score:.4f}")
    return final_score


def main():
    parser = argparse.ArgumentParser(description="Data Cleaning OpenEnv Baseline Inference")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="Base URL of the running environment",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (auto-set to llama-3.3-70b-versatile for groq if not changed)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Task IDs to run (default: 1 2 3)",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "groq"],
        help="LLM provider: openai (paid) or groq (free at console.groq.com)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Set up LLM client
    if args.provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY not set. Get a free key at https://console.groq.com")
            sys.exit(1)
        llm_client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = args.model if args.model != "gpt-4o-mini" else "llama-3.3-70b-versatile"
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set. Get a key at https://platform.openai.com")
            sys.exit(1)
        llm_client = OpenAI(api_key=api_key)
        model = args.model

    print(f"Data Cleaning OpenEnv Baseline")
    print(f"Provider: {args.provider} | Model: {model}")
    print(f"Environment: {args.base_url}")
    print(f"Tasks: {args.tasks}")

    scores = {}
    with DataCleaningEnv(base_url=args.base_url).sync() as env:
        for task_id in args.tasks:
            score = run_task(
                client=llm_client,
                env=env,
                task_id=task_id,
                model=model,
                verbose=args.verbose,
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
