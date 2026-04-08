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

def run_task(env: EnvClient, task_id: int) -> tuple[bool, int, List[float]]:
    """
    Run an LLM agent on one task.
    Returns: (success, step_count, rewards_list)
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
        
        while not done and step_count < 500:
            # Build user message
            user_msg = f"Current dataset state (step {step_count}):\n\nColumns: {column_names}\nTypes: {column_types}\n\nData:\n"
            for i, row in enumerate(current_data):
                user_msg += f"  Row {i}: {row}\n"
            user_msg += f"\nData profile: {json.dumps(data_profile, indent=2)}"
            user_msg += f"\nCurrent quality score: {quality_score:.4f}"
            user_msg += f"\nLast message: {message}"
            user_msg += "\n\nWhat action will you take? Respond with JSON only."
            
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
                print(f"[STEP] step={step_count} action=submit reward=0.00 done=true error={error_msg}")
                rewards.append(0.0)
                break
            
            messages.append({"role": "assistant", "content": assistant_content})
            action = parse_llm_action(assistant_content)
            
            # Step environment
            try:
                result = env.step(action)
                obs = result.get("observation", result)
                
                reward = float(result.get("reward", obs.get("reward", 0.0)))
                done = obs.get("done", False)
                quality_score = obs.get("quality_score", quality_score)
                current_data = obs.get("current_data", current_data)
                data_profile = obs.get("data_profile", data_profile)
                message = obs.get("message", "")
                error_msg = None
                
            except Exception as e:
                reward = 0.0
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
    
    return success, step_count, rewards


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with proper output format."""
    env = EnvClient(base_url=ENV_URL)
    
    try:
        # Wait for environment
        if not wait_for_env(env, max_retries=30, delay=10):
            print(f"[START] task=all env=data-cleaning model={MODEL_NAME}", flush=True)
            print(f"[STEP] step=0 action=none reward=0.00 done=true error=Environment_not_available", flush=True)
            print(f"[END] success=false steps=0 rewards=", flush=True)
            sys.exit(1)
        
        # Run all 3 tasks
        for task_id in [1, 2, 3]:
            task_name = {1: "fix-basics", 2: "clean-customers", 3: "enterprise-reconcile"}[task_id]
            
            # [START] line (required format)
            print(f"[START] task={task_name} env=data-cleaning model={MODEL_NAME}", flush=True)
            
            # Run the task
            success, steps, rewards = run_task(env, task_id)
            
            # [END] line (required format)
            success_str = "true" if success else "false"
            rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
            print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)
    
    except Exception as e:
        err_msg = str(e).replace('\n', ' ').replace('\r', '').replace(' ', '_')[:50]
        print(f"[START] task=all env=data-cleaning model={MODEL_NAME}", flush=True)
        print(f"[STEP] step=0 action=none reward=0.00 done=true error={err_msg}", flush=True)
        print(f"[END] success=false steps=0 rewards=", flush=True)
        sys.exit(1)
    
    finally:
        env.close()


if __name__ == "__main__":
    main()
