"""
Data Cleaning OpenEnv Environment - Baseline Inference Script

OpenEnv Hackathon Submission - Fully compliant with guidelines.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# =============================================================================
# Required Environment Variables (per hackathon guidelines)
# =============================================================================

# Epsilon to ensure scores are strictly within (0, 1) and visible at 2 decimal places
EPSILON = 0.01


def clamp_score(score: float) -> float:
    """Clamp score to be strictly between 0 and 1 (exclusive)."""
    score = max(0.0, min(1.0, score))
    if score <= EPSILON:
        return EPSILON
    if score >= 1.0 - EPSILON:
        return 1.0 - EPSILON
    return score


# API_BASE_URL: LLM API endpoint (must have default)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

# MODEL_NAME: Model identifier for inference (must have default)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# HF_TOKEN: Hugging Face API token (mandatory, no default)
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Environment URL (our HF Space)
ENV_URL = "https://ujjwalml-data-cleaning-env.hf.space"

# =============================================================================
# Initialize OpenAI client using HF_TOKEN as api_key
# =============================================================================

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert data engineer cleaning dirty tabular datasets.

Available actions (respond with ONE JSON object per turn):
- fix_value: Fix a cell value. Params: row (int), col (str), value (str)
- fill_missing: Fill an empty/null cell. Params: row (int), col (str), value (str)  
- delete_row: Remove a duplicate row. Params: row (int)
- fix_type: Fix a type mismatch. Params: row (int), col (str), value (str)
- submit: Finalize your work. No params needed.

Response format (strict JSON, no markdown):
{"action_type": "<action>", "row": <int or null>, "col": "<column_name or null>", "value": "<new_value or null>"}

Data rules:
- Dates: YYYY-MM-DD format, valid month (01-12), valid day (01-31), year 1900-2030
- Emails: must contain exactly one @ with a valid domain (e.g., user@domain.com)
- Integers (age, id): must be positive whole numbers
- Floats (salary, amount): must be positive numbers
- Status: only 'active', 'inactive', or 'on_leave' (lowercase)
- Departments: only 'Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations'

Strategy for maximum score:
1. First scan the data profile for columns with null_count > 0 or type_error_count > 0
2. Fix type errors and fill missing values first (biggest quality impact)
3. Then delete exact duplicate rows
4. Submit once the data profile shows 0 errors across all columns
5. Don't waste steps on cells that are already correct"""


# =============================================================================
# Environment Client
# =============================================================================

class EnvClient:
    """Simple HTTP client for the OpenEnv environment."""
    
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
    
    def reset(self, task_id: int = 1) -> Dict[str, Any]:
        """Reset the environment and start a new episode."""
        response = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Take a step in the environment."""
        response = self.session.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if the environment is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close the session."""
        self.session.close()


# =============================================================================
# Helper Functions
# =============================================================================

def parse_llm_action(content: str) -> Dict[str, Any]:
    """Parse LLM response into an action dictionary."""
    content = content.strip()
    
    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1])
        else:
            content = "\n".join(lines[1:])
    
    try:
        data = json.loads(content)
        return {
            "action_type": data.get("action_type", "submit"),
            "row": data.get("row"),
            "col": data.get("col"),
            "value": data.get("value"),
        }
    except Exception:
        return {"action_type": "submit", "row": None, "col": None, "value": None}


def format_action(action: Dict[str, Any]) -> str:
    """Format action for logging in function-call style."""
    action_type = action.get("action_type", "unknown")
    row = action.get("row")
    col = action.get("col")
    value = action.get("value")
    
    # Format as function call style: action_type(params)
    if action_type == "submit":
        return "submit()"
    elif action_type == "delete_row":
        return f"delete_row({row})"
    elif action_type in ("fix_value", "fill_missing", "fix_type"):
        return f"{action_type}({row},'{col}','{value}')"
    else:
        return f"{action_type}()"


def wait_for_env(env: EnvClient, max_retries: int = 30, delay: int = 10) -> bool:
    """Wait for the environment to become available."""
    for i in range(max_retries):
        if env.health_check():
            return True
        time.sleep(delay)
    return False


# =============================================================================
# Run Task (with proper output format)
# =============================================================================

def run_task(env: EnvClient, task_id: int) -> tuple[bool, int, List[float], float]:
    """
    Run an LLM agent on one task.
    Returns: (success, step_count, rewards_list, final_score)
    """
    rewards: List[float] = []
    step_count = 0
    success = False
    error_msg: Optional[str] = None
    
    try:
        # Reset environment
        result = env.reset(task_id=task_id)
        obs = result.get("observation", result)
        
        quality_score = obs.get("quality_score", 0.0)
        current_data = obs.get("current_data", [])
        column_names = obs.get("column_names", [])
        column_types = obs.get("column_types", [])
        data_profile = obs.get("data_profile", {})
        message = obs.get("message", "")
        done = obs.get("done", False)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        while not done and step_count < 30:
            # Build a compact user message — only send rows with issues, not the full dataset
            user_msg = f"Dataset state (step {step_count}, {len(current_data)} rows):\n"
            user_msg += f"Columns: {column_names}\nExpected types: {column_types}\n"
            user_msg += f"Quality score: {quality_score:.4f}\n"

            # Identify rows with potential issues using actual type checks
            flagged_rows = set()
            for r_idx, row in enumerate(current_data):
                for c_idx, col_name in enumerate(column_names):
                    if c_idx >= len(row):
                        flagged_rows.add(r_idx)
                        continue
                    cell = row[c_idx].strip() if row[c_idx] else ""
                    ctype = column_types[c_idx] if c_idx < len(column_types) else "str"
                    # Check for empty cells
                    if cell == "":
                        flagged_rows.add(r_idx)
                        continue
                    # Check type errors
                    if ctype == "int":
                        try:
                            v = int(cell)
                            if v < 0:
                                flagged_rows.add(r_idx)
                        except ValueError:
                            flagged_rows.add(r_idx)
                    elif ctype == "float":
                        try:
                            v = float(cell)
                            if v <= 0 or v > 500000:
                                flagged_rows.add(r_idx)
                        except ValueError:
                            flagged_rows.add(r_idx)
                    elif ctype == "email":
                        import re
                        if not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", cell):
                            flagged_rows.add(r_idx)
                    elif ctype == "date":
                        import re
                        if not re.match(r"^\d{4}-\d{2}-\d{2}$", cell):
                            flagged_rows.add(r_idx)
                        else:
                            parts = cell.split("-")
                            m, d = int(parts[1]), int(parts[2])
                            yr = int(parts[0])
                            if m < 1 or m > 12 or d < 1 or d > 31 or yr < 1900 or yr > 2030:
                                flagged_rows.add(r_idx)

            # Check for duplicates
            dup_count = data_profile.get("__duplicates__", 0)
            if dup_count > 0:
                seen_rows = {}
                for r_idx, row in enumerate(current_data):
                    key = tuple(row)
                    if key in seen_rows:
                        flagged_rows.add(r_idx)  # flag the duplicate (not the first)
                    else:
                        seen_rows[key] = r_idx

            # On first step, show all data; after that, only flagged rows
            if step_count == 0 or len(flagged_rows) == 0:
                user_msg += "\nAll rows:\n"
                for i, row in enumerate(current_data):
                    user_msg += f"  Row {i}: {row}\n"
            else:
                user_msg += f"\nRows with issues ({len(flagged_rows)} flagged):\n"
                for i in sorted(flagged_rows):
                    if i < len(current_data):
                        user_msg += f"  Row {i}: {current_data[i]}\n"
                # Show 2 clean rows for reference
                clean_shown = 0
                for i, row in enumerate(current_data):
                    if i not in flagged_rows and clean_shown < 2:
                        user_msg += f"  Row {i} (clean ref): {row}\n"
                        clean_shown += 1

            user_msg += f"\nProfile: duplicates={dup_count}"
            for col_name, stats in data_profile.items():
                if col_name == "__duplicates__" or not isinstance(stats, dict):
                    continue
                nulls = stats.get("null_count", 0)
                errs = stats.get("type_error_count", 0)
                if nulls > 0 or errs > 0:
                    user_msg += f", {col_name}(nulls={nulls},type_errors={errs})"

            user_msg += f"\nFeedback: {message}"
            user_msg += f"\nSteps remaining: {30 - step_count}"
            user_msg += "\n\nRespond with ONE JSON action."

            # Sliding window: keep system prompt + last 4 exchanges to avoid context overflow
            if len(messages) > 9:  # system + 4*(user+assistant) = 9
                messages = [messages[0]] + messages[-8:]
            
            messages.append({"role": "user", "content": user_msg})
            
            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=200,
                )
                assistant_content = response.choices[0].message.content or ""
            except Exception as e:
                error_msg = str(e)
                action = {"action_type": "submit"}
                step_count += 1
                clamped_r = clamp_score(0.0)
                print(f"[STEP] step={step_count} action=submit() reward={clamped_r:.2f} done=true error={error_msg}", flush=True)
                rewards.append(clamped_r)
                break
            
            messages.append({"role": "assistant", "content": assistant_content})
            action = parse_llm_action(assistant_content)
            
            # Step environment
            try:
                result = env.step(action)
                obs = result.get("observation", result)
                
                reward = float(result.get("reward", obs.get("reward", 0.0)))
                # Clamp reward to strictly within (0, 1) — validator rejects 0.0 and 1.0
                reward = clamp_score(reward)
                done = obs.get("done", False)
                quality_score = obs.get("quality_score", quality_score)
                current_data = obs.get("current_data", current_data)
                data_profile = obs.get("data_profile", data_profile)
                message = obs.get("message", "")
                error_msg = None
                
            except Exception as e:
                reward = clamp_score(0.0)
                done = True
                error_msg = str(e)
            
            step_count += 1
            rewards.append(reward)
            
            # [STEP] line (required format)
            done_str = "true" if done else "false"
            # Sanitize error message (remove newlines, limit length)
            if error_msg:
                error_str = error_msg.replace('\n', ' ').replace('\r', '')[:100]
            else:
                error_str = "null"
            action_str = format_action(action)
            print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)
        
        # Determine success (quality > 0.5 is a reasonable threshold)
        success = quality_score >= 0.5
        
    except Exception as e:
        error_msg = str(e)
        success = False
    
    final_score = clamp_score(quality_score)
    return success, step_count, rewards, final_score


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with proper output format."""
    env = EnvClient(base_url=ENV_URL)
    
    try:
        # Wait for environment
        if not wait_for_env(env, max_retries=30, delay=10):
            r = clamp_score(0.0)
            print(f"[START] task=unknown env=data-cleaning model={MODEL_NAME}", flush=True)
            print(f"[STEP] step=1 action=submit() reward={r:.2f} done=true error=Environment_not_available", flush=True)
            print(f"[END] success=false steps=1 rewards={r:.2f}", flush=True)
            sys.exit(1)
        
        # Run all 3 tasks
        for task_id in [1, 2, 3]:
            task_name = {1: "fix-basics", 2: "clean-customers", 3: "enterprise-reconcile"}[task_id]
            
            # [START] line (required format)
            print(f"[START] task={task_name} env=data-cleaning model={MODEL_NAME}", flush=True)
            
            # Run the task
            success, steps, rewards, score = run_task(env, task_id)
            
            # [END] line (required format) - rewards already clamped in run_task
            success_str = "true" if success else "false"
            rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else f"{clamp_score(0.0):.2f}"
            print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)
    
    except Exception as e:
        err_msg = str(e).replace('\n', ' ').replace('\r', '').replace(' ', '_')[:50]
        r = clamp_score(0.0)
        print(f"[START] task=unknown env=data-cleaning model={MODEL_NAME}", flush=True)
        print(f"[STEP] step=1 action=submit() reward={r:.2f} done=true error={err_msg}", flush=True)
        print(f"[END] success=false steps=1 rewards={r:.2f}", flush=True)
        sys.exit(1)
    
    finally:
        env.close()


if __name__ == "__main__":
    main()
