"""
Data Cleaning OpenEnv Environment - Graders

Deterministic grader functions for all 3 tasks.
Each grader returns a float strictly within (0.0, 1.0) — never exactly 0 or 1.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

Dataset = List[List[str]]

# Epsilon to ensure scores are strictly within (0, 1) - must be >= 0.01 for 2-decimal formatting
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
    """
    Simple cell-accuracy grader.
    Score = fraction of cells matching ground truth, clamped to (0, 1).
    """
    score = _cell_accuracy(current, ground_truth)
    return round(_clamp_score(score), 4)


# ---------------------------------------------------------------------------
# Task 2 Grader — Medium
# ---------------------------------------------------------------------------

def grade_task_2(current: Dataset, ground_truth: Dataset) -> float:
    """
    Weighted grader:
      40%  Cell accuracy (compared to ground truth with rows properly aligned)
      30%  Duplicate removal score
      30%  Missing value fill score
    """
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
    removed = min(dup_count, expected_dups)  # can't get credit for removing more than injected
    # Actually reward for NOT having duplicates in deduped
    actual_dups_remaining = len(current) - len(deduped)
    dup_score = max(0.0, 1.0 - (actual_dups_remaining / expected_dups))

    # 2. Missing value fill (30 pts)
    missing_cols = {"email": [3, 12], "city": [7], "purchase_amount": [10]}  # row indices with missing values
    filled_count = 0
    total_missing = 4  # 4 missing values injected (rows 0=phone fmt, 3=email, 7=city, 10=amount, 12=email)
    # Check against ground truth rows (using row index within deduped)
    gt_by_id = {row[0]: row for row in ground_truth}
    filled = 0
    total_fillable = 4  # email r3, city r7, amount r10, email r12
    for row in deduped:
        if row[0] in gt_by_id:
            gt_row = gt_by_id[row[0]]
            # If any previously-empty cell is now non-empty and matches gt
            for c in range(len(row)):
                if c < len(gt_row) and row[c].strip() != "" and gt_row[c].strip() != "" and row[c].strip() == gt_row[c].strip():
                    pass  # counted below in cell accuracy

    # Simpler: count missing cells remaining in deduped
    missing_remaining = sum(1 for row in deduped for cell in row if cell.strip() == "")
    fill_score = max(0.0, 1.0 - (missing_remaining / max(total_fillable, 1)))

    # 3. Cell accuracy (40 pts) — compare deduplicated to ground truth
    acc = _cell_accuracy(deduped, ground_truth)

    score = 0.40 * acc + 0.30 * dup_score + 0.30 * fill_score
    return round(_clamp_score(score), 4)


# ---------------------------------------------------------------------------
# Task 3 Grader — Hard
# ---------------------------------------------------------------------------

def grade_task_3(current: Dataset, ground_truth: Dataset) -> float:
    """
    Weighted grader:
      30%  Cell accuracy
      25%  Referential integrity (manager_id must reference valid employee_id)
      25%  Logical consistency (salary > 0, valid status, valid dept, valid date, non-empty name/email)
      20%  Deduplication
    """
    VALID_DEPTS = {"Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"}
    VALID_STATUSES = {"active", "inactive", "on_leave"}

    # Cell accuracy (30%)
    acc = _cell_accuracy(current, ground_truth)

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

        total_logic_checks += 5

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
        if row[6].strip() and not _is_valid_email(row[6]):
            logic_errors += 1
            total_logic_checks += 1
        elif not row[6].strip():
            logic_errors += 1
            total_logic_checks += 1

    logic_score = max(0.0, 1.0 - (logic_errors / max(total_logic_checks, 1)))

    # Deduplication (20%)
    seen = set()
    dups = 0
    for row in current:
        key = tuple(row)
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    dup_score = 1.0 if len(current) == 0 else max(0.0, 1.0 - (dups / len(current)))

    score = 0.30 * acc + 0.25 * ref_score + 0.25 * logic_score + 0.20 * dup_score
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
