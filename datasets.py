"""
Data Cleaning OpenEnv Environment - Dataset Generators

Deterministic dirty + ground-truth datasets for all 3 tasks.
Seeds are fixed so baseline scores are reproducible.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

# Type alias
Dataset = List[List[str]]
TaskData = Tuple[Dataset, Dataset, List[str], List[str]]


# ---------------------------------------------------------------------------
# Task 1 — Easy: Fix obvious type/format errors (10 rows, 4 columns)
# ---------------------------------------------------------------------------

TASK1_COLS = ["name", "age", "email", "signup_date"]
TASK1_TYPES = ["str", "int", "email", "date"]

TASK1_GROUND_TRUTH: Dataset = [
    ["Alice Johnson",  "28", "alice@example.com",    "2023-01-15"],
    ["Bob Smith",      "34", "bob.smith@gmail.com",  "2023-02-20"],
    ["Carol White",    "22", "carol@company.org",    "2023-03-05"],
    ["David Brown",    "45", "david.b@outlook.com",  "2023-04-18"],
    ["Eve Davis",      "31", "eve.davis@yahoo.com",  "2023-05-09"],
    ["Frank Miller",   "27", "frank@example.com",    "2023-06-22"],
    ["Grace Wilson",   "39", "grace.w@gmail.com",    "2023-07-11"],
    ["Henry Moore",    "52", "henry@company.org",    "2023-08-30"],
    ["Iris Taylor",    "24", "iris.t@outlook.com",   "2023-09-14"],
    ["Jack Anderson",  "61", "jack.a@yahoo.com",     "2023-10-03"],
]

TASK1_DIRTY: Dataset = [
    ["Alice Johnson",  "twenty-eight", "alice@example.com",      "2023-01-15"],   # row 0: age wrong type
    ["Bob Smith",      "34",           "bob.smith_at_gmail.com", "2023-02-20"],   # row 1: bad email
    ["Carol White",    "22",           "carol@company.org",      "2023-03-05"],   # row 2: clean
    ["David Brown",    "45",           "david.b@outlook.com",    "20230418"],     # row 3: bad date format
    ["Eve Davis",      "-31",          "eve.davis@yahoo.com",    "2023-05-09"],   # row 4: negative age
    ["Frank Miller",   "27",           "frank@example.com",      "2023-06-22"],   # row 5: clean
    ["Grace Wilson",   "39",           "grace.w_gmail.com",      "2023-07-11"],   # row 6: bad email (missing @)
    ["Henry Moore",    "52",           "henry@company.org",      "2023-08-30"],   # row 7: clean
    ["Iris Taylor",    "24",           "",                       "2023-09-14"],   # row 8: missing email
    ["Jack Anderson",  "61",           "jack.a@yahoo.com",       "2023-13-03"],   # row 9: invalid date month=13
]


def generate_task_1() -> TaskData:
    """Returns (dirty, ground_truth, column_names, column_types)."""
    return (
        copy.deepcopy(TASK1_DIRTY),
        copy.deepcopy(TASK1_GROUND_TRUTH),
        list(TASK1_COLS),
        list(TASK1_TYPES),
    )


# ---------------------------------------------------------------------------
# Task 2 — Medium: Missing values + deduplication (25 rows, 6 columns)
# ---------------------------------------------------------------------------

TASK2_COLS = ["id", "name", "email", "phone", "city", "purchase_amount"]
TASK2_TYPES = ["int", "str", "email", "phone", "str", "float"]

TASK2_GROUND_TRUTH: Dataset = [
    ["1",  "Alice Johnson",   "alice@example.com",    "+1-555-0101", "New York",    "150.00"],
    ["2",  "Bob Smith",       "bob@example.com",      "+1-555-0102", "Los Angeles", "89.99"],
    ["3",  "Carol White",     "carol@example.com",    "+1-555-0103", "Chicago",     "220.50"],
    ["4",  "David Brown",     "david@example.com",    "+1-555-0104", "Houston",     "45.00"],
    ["5",  "Eve Davis",       "eve@example.com",      "+1-555-0105", "Phoenix",     "310.75"],
    ["6",  "Frank Miller",    "frank@example.com",    "+1-555-0106", "Philadelphia","99.00"],
    ["7",  "Grace Wilson",    "grace@example.com",    "+1-555-0107", "San Antonio", "175.25"],
    ["8",  "Henry Moore",     "henry@example.com",    "+1-555-0108", "San Diego",   "60.00"],
    ["9",  "Iris Taylor",     "iris@example.com",     "+1-555-0109", "Dallas",      "420.00"],
    ["10", "Jack Anderson",   "jack@example.com",     "+1-555-0110", "San Jose",    "35.50"],
    ["11", "Karen Thomas",    "karen@example.com",    "+1-555-0111", "Austin",      "280.00"],
    ["12", "Leo Jackson",     "leo@example.com",      "+1-555-0112", "Jacksonville","115.00"],
    ["13", "Mia Harris",      "mia@example.com",      "+1-555-0113", "Fort Worth",  "90.25"],
    ["14", "Noah Martin",     "noah@example.com",     "+1-555-0114", "Columbus",    "195.00"],
    ["15", "Olivia Garcia",   "olivia@example.com",   "+1-555-0115", "Charlotte",   "55.00"],
    ["16", "Paul Martinez",   "paul@example.com",     "+1-555-0116", "Indianapolis","340.50"],
    ["17", "Quinn Robinson",  "quinn@example.com",    "+1-555-0117", "San Francisco","210.00"],
    ["18", "Rachel Clark",    "rachel@example.com",   "+1-555-0118", "Seattle",     "75.00"],
    ["19", "Sam Rodriguez",   "sam@example.com",      "+1-555-0119", "Denver",      "160.00"],
    ["20", "Tina Lewis",      "tina@example.com",     "+1-555-0120", "Nashville",   "480.00"],
]

TASK2_DIRTY: Dataset = [
    ["1",  "Alice Johnson",   "alice@example.com",    "555-0101",       "New York",    "150.00"],   # bad phone format
    ["2",  "Bob Smith",       "bob@example.com",      "+1-555-0102",    "Los Angeles", "89.99"],
    ["3",  "Carol White",     "carol@example.com",    "+1-555-0103",    "Chicago",     "220.50"],
    ["4",  "David Brown",     "",                     "+1-555-0104",    "Houston",     "45.00"],    # missing email
    ["5",  "Eve Davis",       "eve@example.com",      "+1-555-0105",    "Phoenix",     "310.75"],
    ["6",  "Frank Miller",    "frank@example.com",    "+1-555-0106",    "Philadelphia","99.00"],
    ["7",  "Grace Wilson",    "grace@example.com",    "+1-555-0107",    "San Antonio", "175.25"],
    ["8",  "Henry Moore",     "henry@example.com",    "+1-555-0108",    "",            "60.00"],    # missing city
    ["9",  "Iris Taylor",     "iris@example.com",     "+1-555-0109",    "Dallas",      "420.00"],
    ["10", "Jack Anderson",   "jack@example.com",     "+1-555-0110",    "San Jose",    "35.50"],
    ["11", "Karen Thomas",    "karen@example.com",    "+1-555-0111",    "Austin",      ""],         # missing amount
    ["12", "Leo Jackson",     "leo@example.com",      "+1-555-0112",    "Jacksonville","115.00"],
    ["13", "Mia Harris",      "",                     "+1-555-0113",    "Fort Worth",  "90.25"],    # missing email
    ["14", "Noah Martin",     "noah@example.com",     "+1-555-0114",    "Columbus",    "195.00"],
    ["15", "Olivia Garcia",   "olivia@example.com",   "+1-555-0115",    "Charlotte",   "55.00"],
    ["16", "Paul Martinez",   "paul@example.com",     "+1-555-0116",    "Indianapolis","340.50"],
    ["17", "Quinn Robinson",  "quinn@example.com",    "+1-555-0117",    "San Francisco","210.00"],
    ["18", "Rachel Clark",    "rachel@example.com",   "+1-555-0118",    "Seattle",     "75.00"],
    ["19", "Sam Rodriguez",   "sam@example.com",      "+1-555-0119",    "Denver",      "160.00"],
    ["20", "Tina Lewis",      "tina@example.com",     "+1-555-0120",    "Nashville",   "480.00"],
    # Duplicates below
    ["2",  "Bob Smith",       "bob@example.com",      "+1-555-0102",    "Los Angeles", "89.99"],    # exact dup
    ["5",  "Eve Davis",       "eve@example.com",      "+1-555-0105",    "Phoenix",     "310.75"],   # exact dup
    ["9",  "Iris Taylor",     "iris@example.com",     "+1-555-0109",    "Dallas",      "420.00"],   # exact dup
    ["14", "Noah Martin",     "noah@example.com",     "+1-555-0114",    "Columbus",    "195.00"],   # exact dup
    ["17", "Quinn Robinson",  "quinn@example.com",    "+1-555-0117",    "San Francisco","210.00"],  # exact dup
]


def generate_task_2() -> TaskData:
    """Returns (dirty, ground_truth, column_names, column_types)."""
    return (
        copy.deepcopy(TASK2_DIRTY),
        copy.deepcopy(TASK2_GROUND_TRUTH),
        list(TASK2_COLS),
        list(TASK2_TYPES),
    )


# ---------------------------------------------------------------------------
# Task 3 — Hard: Enterprise cross-column reconciliation (50 rows, 8 columns)
# ---------------------------------------------------------------------------

TASK3_COLS = [
    "employee_id", "name", "department", "salary",
    "manager_id", "start_date", "email", "status"
]
TASK3_TYPES = ["int", "str", "str", "float", "int", "date", "email", "str"]

VALID_DEPTS = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]
VALID_STATUSES = ["active", "inactive", "on_leave"]

TASK3_GROUND_TRUTH: Dataset = [
    ["1",  "Mary Adams",      "Engineering", "95000.00", "0",  "2018-03-12", "mary.adams@corp.com",      "active"],
    ["2",  "James Baker",     "Marketing",   "72000.00", "1",  "2019-06-01", "james.baker@corp.com",     "active"],
    ["3",  "Linda Campbell",  "Sales",       "65000.00", "1",  "2020-01-15", "linda.campbell@corp.com",  "active"],
    ["4",  "Robert Davis",    "HR",          "68000.00", "1",  "2017-09-20", "robert.davis@corp.com",    "active"],
    ["5",  "Patricia Evans",  "Finance",     "80000.00", "1",  "2016-11-03", "patricia.evans@corp.com",  "active"],
    ["6",  "Charles Foster",  "Engineering", "88000.00", "1",  "2019-04-22", "charles.foster@corp.com",  "active"],
    ["7",  "Barbara Green",   "Marketing",   "61000.00", "2",  "2021-02-18", "barbara.green@corp.com",   "active"],
    ["8",  "Thomas Harris",   "Sales",       "59000.00", "3",  "2022-07-05", "thomas.harris@corp.com",   "active"],
    ["9",  "Jessica Ingram",  "HR",          "64000.00", "4",  "2020-10-30", "jessica.ingram@corp.com",  "active"],
    ["10", "William Johnson", "Finance",     "91000.00", "5",  "2015-05-14", "william.johnson@corp.com", "active"],
    ["11", "Sarah King",      "Operations",  "55000.00", "1",  "2023-01-09", "sarah.king@corp.com",      "active"],
    ["12", "Joseph Lee",      "Engineering", "82000.00", "6",  "2018-08-27", "joseph.lee@corp.com",      "active"],
    ["13", "Karen Martinez",  "Marketing",   "70000.00", "2",  "2019-12-11", "karen.martinez@corp.com",  "active"],
    ["14", "Donald Nelson",   "Sales",       "63000.00", "3",  "2021-05-23", "donald.nelson@corp.com",   "inactive"],
    ["15", "Lisa Oliver",     "HR",          "66000.00", "4",  "2020-03-07", "lisa.oliver@corp.com",     "active"],
    ["16", "Mark Parker",     "Finance",     "77000.00", "5",  "2017-07-19", "mark.parker@corp.com",     "active"],
    ["17", "Nancy Quinn",     "Operations",  "52000.00", "1",  "2022-11-28", "nancy.quinn@corp.com",     "on_leave"],
    ["18", "Paul Roberts",    "Engineering", "93000.00", "6",  "2016-02-14", "paul.roberts@corp.com",    "active"],
    ["19", "Sandra Scott",    "Marketing",   "68000.00", "2",  "2020-09-03", "sandra.scott@corp.com",    "active"],
    ["20", "Kevin Turner",    "Sales",       "61000.00", "3",  "2021-08-16", "kevin.turner@corp.com",    "active"],
    ["21", "Betty Underwood", "HR",          "72000.00", "4",  "2018-04-25", "betty.underwood@corp.com", "active"],
    ["22", "George Vance",    "Finance",     "85000.00", "5",  "2019-01-08", "george.vance@corp.com",    "active"],
    ["23", "Helen Walker",    "Operations",  "58000.00", "1",  "2023-03-21", "helen.walker@corp.com",    "active"],
    ["24", "Steven Xavier",   "Engineering", "79000.00", "6",  "2020-06-10", "steven.xavier@corp.com",   "active"],
    ["25", "Donna Young",     "Marketing",   "65000.00", "2",  "2021-10-04", "donna.young@corp.com",     "inactive"],
    ["26", "Edward Zimmer",   "Sales",       "57000.00", "3",  "2022-02-28", "edward.zimmer@corp.com",   "active"],
    ["27", "Ruth Abbott",     "HR",          "69000.00", "4",  "2019-07-17", "ruth.abbott@corp.com",     "active"],
    ["28", "Arthur Burton",   "Finance",     "83000.00", "5",  "2016-12-06", "arthur.burton@corp.com",   "active"],
    ["29", "Frances Carr",    "Operations",  "54000.00", "1",  "2022-05-13", "frances.carr@corp.com",    "on_leave"],
    ["30", "Harold Dean",     "Engineering", "97000.00", "6",  "2015-09-29", "harold.dean@corp.com",     "active"],
    ["31", "Deborah Ellis",   "Marketing",   "73000.00", "2",  "2018-11-22", "deborah.ellis@corp.com",   "active"],
    ["32", "Jack Fitzgerald", "Sales",       "60000.00", "3",  "2020-04-08", "jack.fitzgerald@corp.com", "active"],
    ["33", "Carol Golden",    "HR",          "67000.00", "4",  "2021-09-15", "carol.golden@corp.com",    "active"],
    ["34", "Walter Hamilton", "Finance",     "89000.00", "5",  "2017-03-27", "walter.hamilton@corp.com", "active"],
    ["35", "Catherine Irwin", "Operations",  "53000.00", "1",  "2023-06-05", "catherine.irwin@corp.com", "active"],
    ["36", "Ralph Jenkins",   "Engineering", "86000.00", "6",  "2019-08-19", "ralph.jenkins@corp.com",   "active"],
    ["37", "Maria Klein",     "Marketing",   "71000.00", "2",  "2020-12-30", "maria.klein@corp.com",     "active"],
    ["38", "Russell Long",    "Sales",       "62000.00", "3",  "2021-06-24", "russell.long@corp.com",    "inactive"],
    ["39", "Sharon Mills",    "HR",          "65000.00", "4",  "2018-02-11", "sharon.mills@corp.com",    "active"],
    ["40", "Peter Norton",    "Finance",     "78000.00", "5",  "2016-10-16", "peter.norton@corp.com",    "active"],
    ["41", "Angela Owen",     "Operations",  "56000.00", "1",  "2022-08-02", "angela.owen@corp.com",     "active"],
    ["42", "Douglas Porter",  "Engineering", "84000.00", "6",  "2017-05-07", "douglas.porter@corp.com",  "active"],
    ["43", "Evelyn Quinn",    "Marketing",   "67000.00", "2",  "2019-03-25", "evelyn.quinn@corp.com",    "active"],
    ["44", "Frank Riley",     "Sales",       "58000.00", "3",  "2021-11-18", "frank.riley@corp.com",     "on_leave"],
    ["45", "Gloria Sanders",  "HR",          "71000.00", "4",  "2018-07-09", "gloria.sanders@corp.com",  "active"],
    ["46", "Howard Thompson", "Finance",     "92000.00", "5",  "2015-12-21", "howard.thompson@corp.com", "active"],
    ["47", "Irene Underhill", "Operations",  "51000.00", "1",  "2023-04-14", "irene.underhill@corp.com", "active"],
    ["48", "Jerry Valentine", "Engineering", "81000.00", "6",  "2020-07-31", "jerry.valentine@corp.com", "active"],
    ["49", "Kathleen Warren", "Marketing",   "69000.00", "2",  "2019-10-20", "kathleen.warren@corp.com", "active"],
    ["50", "Lawrence Xavier", "Sales",       "64000.00", "3",  "2022-01-06", "lawrence.xavier@corp.com", "active"],
]

def _make_task3_dirty() -> Dataset:
    """Inject 25 errors into Task 3 ground truth."""
    dirty = copy.deepcopy(TASK3_GROUND_TRUTH)
    # Row 0 (Mary Adams) is the CEO — manager_id 0 is valid
    # Error 1: negative salary
    dirty[1][3] = "-72000.00"
    # Error 2: invalid department
    dirty[2][2] = "Legals"
    # Error 3: bad email
    dirty[3][6] = "robert.davis_at_corp.com"
    # Error 4: invalid status
    dirty[4][7] = "retired"
    # Error 5: future start_date
    dirty[5][5] = "2035-04-22"
    # Error 6: manager_id references non-existent employee (999)
    dirty[6][4] = "999"
    # Error 7: manager_id references non-existent employee (998)
    dirty[7][4] = "998"
    # Error 8: bad date format
    dirty[8][5] = "10/30/2020"
    # Error 9: salary as text
    dirty[9][3] = "ninety-one thousand"
    # Error 10: missing email
    dirty[10][6] = ""
    # Error 11: manager self-referencing (manager_id == employee_id)
    dirty[11][4] = "12"
    # Error 12: missing start_date
    dirty[12][5] = ""
    # Error 13: invalid status typo
    dirty[13][7] = "Inactive"  # should be lowercase "inactive"
    # Error 14: negative salary
    dirty[14][3] = "-66000.00"
    # Error 15: bad email (double @)
    dirty[15][6] = "mark@@parker@corp.com"
    # Error 16: wrong department casing
    dirty[16][2] = "operations"  # should be "Operations"
    # Error 17: missing name
    dirty[17][1] = ""
    # Error 18: salary above reasonable ceiling (>500k for non-executives)
    dirty[18][3] = "68000000.00"
    # Error 19: invalid date
    dirty[19][5] = "2021-13-16"
    # Error 20: manager_id pointing to non-existent 997
    dirty[20][4] = "997"
    # Error 21: missing department
    dirty[21][2] = ""
    # Error 22: bad phone — this col doesn't exist but email has extra space
    dirty[22][6] = " helen.walker@corp.com"
    # Error 23: salary zero
    dirty[23][3] = "0.00"
    # Duplicate row: add row 24 as exact copy of row 24
    # Error 24: exact duplicate of row index 0 (Mary Adams) inserted as row 25
    dirty[24] = copy.deepcopy(dirty[0])   # Donna Young replaced by Mary Adams dup — agent must restore
    # Actually let's just make it a near-duplicate name error
    dirty[24][1] = "Donna Young"  # keep name, but make email mismatch
    dirty[24][6] = "donna.young_corp.com"  # bad email
    # Error 25: status empty
    dirty[25][7] = ""

    return dirty

TASK3_DIRTY: Dataset = _make_task3_dirty()


def generate_task_3() -> TaskData:
    """Returns (dirty, ground_truth, column_names, column_types)."""
    return (
        copy.deepcopy(TASK3_DIRTY),
        copy.deepcopy(TASK3_GROUND_TRUTH),
        list(TASK3_COLS),
        list(TASK3_TYPES),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    1: generate_task_1,
    2: generate_task_2,
    3: generate_task_3,
}

TASK_MAX_STEPS = {
    1: 15,
    2: 30,
    3: 60,
}

TASK_DESCRIPTIONS = {
    1: "Fix obvious type and format errors in a small 10-row dataset.",
    2: "Handle missing values and remove duplicate rows in a 25-row customer list.",
    3: "Resolve cross-column inconsistencies including referential integrity, logical violations, "
       "and subtle errors in a 50-row enterprise employee dataset.",
}
