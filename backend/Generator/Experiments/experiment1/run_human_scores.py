"""
Experiment 1 — human-annotation baseline tables.

Reads the 60 annotated posts from the Excel file and computes the same
summary statistics as experiment_2/run.py, but using human expert scores
instead of LLM judge scores.

Post order in the Excel (rows 0–59):
  rows  0–14  → ZS-Pre
  rows 15–29  → FS-Pre
  rows 30–44  → ZS-Post
  rows 45–59  → FS-Post

Table 1 (6 × 4) — mean human dimension score per condition
Table 2 (3 × 4) — proportion of posts flagged for each violation type per condition
"""

from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

ANNOTATION_FILE = "backend/Generator/Experiments/data/UvA Expert Voice - Output annotation.xlsx"
RESULTS_DIR     = Path("backend/Generator/Experiments/experiment1/results")

CONDITION_SLICES = {
    "ZS-Pre":  (0,  15),
    "FS-Pre":  (15, 30),
    "ZS-Post": (30, 45),
    "FS-Post": (45, 60),
}

DIMENSIONS = [
    "tone_of_voice",
    "language_and_style",
    "coherence_and_readability",
    "discourse_structure",
    "specificity",
    "historical_similarity",
]

VIOLATION_TYPES = ["contrastive", "from_to", "this_that"]
VIOL_COL = "discourse violation type  [contrastive], [from-to],  [this-that] or [None]"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_excel(ANNOTATION_FILE, header=0)
    df.columns = [c.strip() for c in df.columns]
    return df.iloc[:60].reset_index(drop=True)  # rows 0–59 only


def parse_violations(raw: str) -> dict[str, bool]:
    s = str(raw).lower()
    return {
        "contrastive": "contrastive" in s,
        "from_to":     "from-to" in s or "from_to" in s,
        "this_that":   "this-that" in s or "this_that" in s or "vague" in s,
    }


# ── Table builders ────────────────────────────────────────────────────────────

def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    """6 × 4 mean human dimension scores per condition."""
    data: dict[str, dict[str, float]] = {}
    for cond, (start, end) in CONDITION_SLICES.items():
        chunk = df.iloc[start:end]
        data[cond] = {
            dim: round(float(chunk[dim].mean()), 2)
            for dim in DIMENSIONS
        }
    result = pd.DataFrame(data, index=DIMENSIONS)
    result.index.name = "Dimension"
    return result[list(CONDITION_SLICES.keys())]


def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    """3 × 4 proportion of posts containing each violation type per condition."""
    data: dict[str, dict[str, float]] = {}
    for cond, (start, end) in CONDITION_SLICES.items():
        chunk = df.iloc[start:end]
        n = len(chunk)
        counts: dict[str, int] = {vt: 0 for vt in VIOLATION_TYPES}
        for val in chunk[VIOL_COL]:
            flags = parse_violations(str(val))
            for vt, flagged in flags.items():
                if flagged:
                    counts[vt] += 1
        data[cond] = {vt: round(counts[vt] / n, 2) for vt in VIOLATION_TYPES}
    result = pd.DataFrame(data, index=VIOLATION_TYPES)
    result.index.name = "Violation Type"
    return result[list(CONDITION_SLICES.keys())]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} annotated posts from {ANNOTATION_FILE}\n")

    table1 = build_table1(df)
    table2 = build_table2(df)

    print("── Table 1: Mean human scores per dimension ────────────────────")
    print(table1.to_string())
    print()
    print("── Table 2: Violation proportions (human) ──────────────────────")
    print(table2.to_string())

    t1 = RESULTS_DIR / "table_human_mean_scores.xlsx"
    t2 = RESULTS_DIR / "table_human_violations.xlsx"
    table1.to_excel(t1)
    table2.to_excel(t2)
    print(f"\nTable 1 -> {t1}")
    print(f"Table 2 -> {t2}")


if __name__ == "__main__":
    main()

# python3 -m backend.Generator.Experiments.experiment1.run_human_scores
