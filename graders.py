"""
Data Cleaning OpenEnv Environment - Graders

Deterministic grader functions for all 3 tasks.
Each grader returns a float strictly within (0.0, 1.0) — never exactly 0 or 1.

Grading philosophy (per OpenEnv best practices):
  - Partial credit for near-miss values (fuzzy matching)
  - Smoother reward signal for RL training
  - Deterministic and reproducible
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

Dataset = List[List[str]]

# Epsilon to ensure scores are strictly within (0, 1)
EPSILON = 0.01


def _clamp_score(score: float) -> float:
    """Clamp score to be strictly between 0 and 1 (exclusive)."""
    score = max(0.0, min(1.0, score))
    if score <= EPSILON:
        return EPSILON
    if score >= 1.0 - EPSILON:
        return 1.0 - EPSILON
    return score


# ---------------------------------------------------------------------------
# Fuzzy matching utilities (inspired by FinQA rewards.py)
# ---------------------------------------------------------------------------

def _fuzzy_cell_score(current_val: str, gt_val: str, col_type: str = "str") -> float:
    """Compute a similarity score between a cell value and ground truth.

    Returns 1.0 for exact match, partial credit for near-misses, 0.0 for
    completely wrong values. This provides a smoother reward signal for RL
    training compared to binary matching.

    Args:
        current_val: The current cell value.
        gt_val: The ground-truth cell value.
        col_type: Expected column type (int, float, str, date, email).

    Returns:
        Score between 0.0 and 1.0.
    """
    cv = current_val.strip()
    gv = gt_val.strip()

    # Exact match
    if cv == gv:
        return 1.0

    # Empty vs non-empty
    if not cv or not gv:
        return 0.0

    # Type-specific fuzzy matching
    if col_type == "int":
        return _fuzzy_int(cv, gv)
    elif col_type == "float":
        return _fuzzy_float(cv, gv)
    elif col_type == "date":
        return _fuzzy_date(cv, gv)
    elif col_type == "email":
        return _fuzzy_email(cv, gv)
    else:
        return _fuzzy_string(cv, gv)


def _fuzzy_int(cv: str, gv: str) -> float:
    """Fuzzy match for integer values. Tolerates small differences."""
    try:
        ci, gi = int(cv), int(gv)
        if ci == gi:
            return 1.0
        # Within 1 → 80% credit (e.g., age off by 1)
        if abs(ci - gi) <= 1:
            return 0.8
        # Within 10% → 50% credit
        if gi != 0 and abs(ci - gi) / abs(gi) <= 0.1:
            return 0.5
        return 0.0
    except ValueError:
        return 0.0


def _fuzzy_float(cv: str, gv: str) -> float:
    """Fuzzy match for float values. Uses relative tolerance."""
    try:
        cf, gf = float(cv), float(gv)
        if cf == gf:
            return 1.0
        if gf == 0:
            return 0.0 if abs(cf) > 0.01 else 1.0
        relative_error = abs(cf - gf) / abs(gf)
        if relative_error <= 0.01:
            return 0.95  # within 1% → near-perfect
        elif relative_error <= 0.05:
            return 0.7   # within 5% → good
        elif relative_error <= 0.1:
            return 0.5   # within 10% → partial
        return 0.0
    except ValueError:
        return 0.0


def _fuzzy_date(cv: str, gv: str) -> float:
    """Fuzzy match for dates. Partial credit for correct year/month."""
    # Try to parse YYYY-MM-DD
    cv_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", cv.strip())
    gv_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", gv.strip())

    if not cv_match or not gv_match:
        # Format doesn't match — try alternate formats for partial credit
        # e.g., 2024/01/15 vs 2024-01-15
        cv_normalized = cv.replace("/", "-").replace(".", "-").strip()
        if cv_normalized == gv.strip():
            return 0.9  # correct date, wrong separator
        return 0.0

    cy, cm, cd = int(cv_match.group(1)), int(cv_match.group(2)), int(cv_match.group(3))
    gy, gm, gd = int(gv_match.group(1)), int(gv_match.group(2)), int(gv_match.group(3))

    if cy == gy and cm == gm and cd == gd:
        return 1.0
    elif cy == gy and cm == gm:
        return 0.7  # right year and month, wrong day
    elif cy == gy:
        return 0.4  # right year only
    return 0.0


def _fuzzy_email(cv: str, gv: str) -> float:
    """Fuzzy match for email addresses. Partial credit for correct domain."""
    cv_lower = cv.strip().lower()
    gv_lower = gv.strip().lower()

    if cv_lower == gv_lower:
        return 1.0

    # Check valid email format
    email_pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, cv_lower):
        return 0.0

    # Split into local + domain
    if "@" in cv_lower and "@" in gv_lower:
        cv_local, cv_domain = cv_lower.rsplit("@", 1)
        gv_local, gv_domain = gv_lower.rsplit("@", 1)
        if cv_domain == gv_domain:
            # Right domain, wrong local part → partial credit
            local_sim = SequenceMatcher(None, cv_local, gv_local).ratio()
            return 0.5 + 0.4 * local_sim  # 0.5–0.9 range
    return 0.1  # valid email but completely wrong


def _fuzzy_string(cv: str, gv: str) -> float:
    """Fuzzy match for generic strings using sequence similarity."""
    cv_lower = cv.strip().lower()
    gv_lower = gv.strip().lower()

    if cv_lower == gv_lower:
        return 1.0

    similarity = SequenceMatcher(None, cv_lower, gv_lower).ratio()
    # Only give credit for high similarity (>0.6)
    if similarity >= 0.9:
        return 0.9
    elif similarity >= 0.8:
        return 0.7
    elif similarity >= 0.6:
        return 0.4
    return 0.0


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _cell_accuracy(current: Dataset, ground_truth: Dataset) -> float:
    """Fraction of cells that exactly match ground truth (ignoring extra/missing rows)."""
    if not ground_truth:
        return 1.0
    total_cells = sum(len(row) for row in ground_truth)
    if total_cells == 0:
        return 1.0

    correct = 0
    for r_idx, gt_row in enumerate(ground_truth):
        if r_idx >= len(current):
            break
        for c_idx, gt_val in enumerate(gt_row):
            if c_idx < len(current[r_idx]):
                if current[r_idx][c_idx].strip() == gt_val.strip():
                    correct += 1

    return correct / total_cells


def _fuzzy_cell_accuracy(
    current: Dataset,
    ground_truth: Dataset,
    col_types: List[str] | None = None,
) -> float:
    """Fuzzy cell accuracy with partial credit for near-miss values.

    Returns a score between 0 and 1 where partial matches contribute
    proportionally based on their similarity to ground truth.
    """
    if not ground_truth:
        return 1.0
    total_cells = sum(len(row) for row in ground_truth)
    if total_cells == 0:
        return 1.0

    score_sum = 0.0
    for r_idx, gt_row in enumerate(ground_truth):
        if r_idx >= len(current):
            break
        for c_idx, gt_val in enumerate(gt_row):
            if c_idx < len(current[r_idx]):
                col_type = col_types[c_idx] if col_types and c_idx < len(col_types) else "str"
                score_sum += _fuzzy_cell_score(current[r_idx][c_idx], gt_val, col_type)

    return score_sum / total_cells


def _is_valid_email(val: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, val.strip()))


def _is_valid_date(val: str) -> bool:
    """Accepts YYYY-MM-DD format with valid month/day ranges."""
    pattern = r"^\d{4}-\d{2}-\d{2}$"
    if not re.match(pattern, val.strip()):
        return False
    try:
        parts = val.strip().split("-")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        if month < 1 or month > 12:
            return False
        if day < 1 or day > 31:
            return False
        # Year sanity (1900–2030)
        if year < 1900 or year > 2030:
            return False
        return True
    except Exception:
        return False


def _is_valid_int(val: str) -> bool:
    try:
        int(val.strip())
        return True
    except ValueError:
        return False


def _is_positive_float(val: str) -> bool:
    try:
        return float(val.strip()) >= 0
    except ValueError:
        return False


def _duplicate_fraction(data: Dataset) -> float:
    """Fraction of rows that are duplicates (i.e., excess beyond first occurrence)."""
    if not data:
        return 0.0
    seen = set()
    dupes = 0
    for row in data:
        key = tuple(row)
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
    return dupes / len(data)


# ---------------------------------------------------------------------------
# Task 1 Grader — Easy
# ---------------------------------------------------------------------------

def grade_task_1(current: Dataset, ground_truth: Dataset) -> float:
    """Grader for Task 1 (easy): Fix the Basics.

    Uses a blend of exact cell accuracy (70%) and fuzzy cell accuracy (30%).
    The fuzzy component provides smoother reward signal by giving partial
    credit for near-miss values (e.g., right date format, wrong day).

    Score components:
      - 70% exact cell accuracy
      - 30% fuzzy cell accuracy (with type-aware matching)
    """
    col_types = ["str", "int", "email", "date"]  # name, age, email, signup_date
    exact = _cell_accuracy(current, ground_truth)
    fuzzy = _fuzzy_cell_accuracy(current, ground_truth, col_types)
    score = 0.70 * exact + 0.30 * fuzzy
    return round(_clamp_score(score), 4)


# ---------------------------------------------------------------------------
# Task 2 Grader — Medium
# ---------------------------------------------------------------------------

def grade_task_2(current: Dataset, ground_truth: Dataset) -> float:
    """Grader for Task 2 (medium): Clean the Customer List.

    Weighted grader:
      35%  Cell accuracy (exact match, compared after deduplication)
      15%  Fuzzy cell accuracy (partial credit for near values)
      30%  Duplicate removal score
      20%  Missing value fill score
    """
    col_types = ["int", "str", "email", "str", "str", "float"]  # id, name, email, phone, city, amount

    # Deduplicate first for fair comparison
    seen: Dict[Tuple, int] = {}
    deduped: Dataset = []
    dup_count = 0
    for row in current:
        key = tuple(row)
        if key not in seen:
            seen[key] = 1
            deduped.append(row)
        else:
            dup_count += 1

    # 1. Duplicate removal (30 pts)
    expected_dups = 5  # we injected 5 duplicates
    actual_dups_remaining = len(current) - len(deduped)
    dup_score = max(0.0, 1.0 - (actual_dups_remaining / expected_dups))

    # 2. Missing value fill (20 pts)
    total_fillable = 4  # 4 missing values injected
    missing_remaining = sum(1 for row in deduped for cell in row if cell.strip() == "")
    fill_score = max(0.0, 1.0 - (missing_remaining / max(total_fillable, 1)))

    # 3. Exact cell accuracy (35 pts)
    exact_acc = _cell_accuracy(deduped, ground_truth)

    # 4. Fuzzy cell accuracy (15 pts)
    fuzzy_acc = _fuzzy_cell_accuracy(deduped, ground_truth, col_types)

    score = 0.35 * exact_acc + 0.15 * fuzzy_acc + 0.30 * dup_score + 0.20 * fill_score
    return round(_clamp_score(score), 4)


# ---------------------------------------------------------------------------
# Task 3 Grader — Hard
# ---------------------------------------------------------------------------

def grade_task_3(current: Dataset, ground_truth: Dataset) -> float:
    """Grader for Task 3 (hard): Enterprise Data Reconciliation.

    Weighted grader:
      25%  Exact cell accuracy
      10%  Fuzzy cell accuracy (partial credit for near-miss values)
      25%  Referential integrity (manager_id must reference valid employee_id)
      25%  Logical consistency (salary > 0, valid status, dept, date, name, email)
      15%  Deduplication
    """
    col_types = ["int", "str", "str", "float", "int", "date", "email", "str"]
    VALID_DEPTS = {"Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"}
    VALID_STATUSES = {"active", "inactive", "on_leave"}

    # Exact cell accuracy (25%)
    exact_acc = _cell_accuracy(current, ground_truth)

    # Fuzzy cell accuracy (10%)
    fuzzy_acc = _fuzzy_cell_accuracy(current, ground_truth, col_types)

    # Referential integrity (25%)
    employee_ids = set()
    for row in current:
        if row and _is_valid_int(row[0]):
            employee_ids.add(row[0].strip())

    referential_errors = 0
    total_ref_checks = 0
    for row in current:
        if len(row) < 5:
            continue
        mgr_id = row[4].strip()
        if mgr_id == "0":  # CEO has no manager
            continue
        total_ref_checks += 1
        if mgr_id not in employee_ids:
            referential_errors += 1

    ref_score = 1.0 if total_ref_checks == 0 else max(0.0, 1.0 - (referential_errors / max(total_ref_checks, 1)))

    # Logical consistency (25%)
    logic_errors = 0
    total_logic_checks = 0
    for row in current:
        if len(row) < 8:
            logic_errors += 8
            total_logic_checks += 8
            continue

        total_logic_checks += 6  # 6 checks per valid row

        # salary > 0
        if not _is_positive_float(row[3]) or float(row[3]) <= 0:
            logic_errors += 1
        elif float(row[3]) > 500000:  # unreasonable salary ceiling
            logic_errors += 1

        # valid status
        if row[7].strip() not in VALID_STATUSES:
            logic_errors += 1

        # valid department
        if row[2].strip() not in VALID_DEPTS:
            logic_errors += 1

        # valid date
        if not _is_valid_date(row[5]):
            logic_errors += 1

        # non-empty name
        if not row[1].strip():
            logic_errors += 1

        # valid email
        if not row[6].strip() or not _is_valid_email(row[6]):
            logic_errors += 1

    logic_score = max(0.0, 1.0 - (logic_errors / max(total_logic_checks, 1)))

    # Deduplication (15%)
    seen = set()
    dups = 0
    for row in current:
        key = tuple(row)
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    dup_score = 1.0 if len(current) == 0 else max(0.0, 1.0 - (dups / len(current)))

    score = (
        0.25 * exact_acc
        + 0.10 * fuzzy_acc
        + 0.25 * ref_score
        + 0.25 * logic_score
        + 0.15 * dup_score
    )
    return round(_clamp_score(score), 4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADERS = {
    1: grade_task_1,
    2: grade_task_2,
    3: grade_task_3,
}


def grade(task_id: int, current: Dataset, ground_truth: Dataset) -> float:
    """Grade a dataset for the given task_id. Returns score in [0.0, 1.0]."""
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(GRADERS.keys())}")
    return GRADERS[task_id](current, ground_truth)
