"""
============================================================
  Student Performance Analytics Dashboard
  Standalone Data Analysis Script
  File: analysis.py

  Purpose
  -------
  This script performs the complete data analysis pipeline
  for the Students Performance in Exams dataset (Kaggle).
  It is independent of the web application and can be run
  on its own to reproduce all key findings.

  Usage
  -----
      python analysis.py

  Requirements
  ------------
      pip install pandas numpy

  Dataset
  -------
  StudentsPerformance.csv (Kaggle — Students Performance in Exams)
  Expected in the same directory as this script.
============================================================
"""

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "StudentsPerformance.csv")

# Pass/Fail threshold (average score)
PASS_THRESHOLD = 60

# Performance category thresholds (based on average score)
# These are analytical categories defined for this project —
# they are NOT official school grading standards.
PERFORMANCE_THRESHOLDS = {
    "Excellent":         (90, 100),
    "Good":              (75, 89),
    "Average":           (60, 74),
    "Needs Improvement": (0,  59),
}

DIVIDER = "-" * 60


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load the CSV file and return a raw DataFrame."""
    print(f"\n{'=' * 60}")
    print("  STEP 1: LOADING DATA")
    print(f"{'=' * 60}")

    df = pd.read_csv(path)

    print(f"  File       : {path}")
    print(f"  Rows       : {len(df)}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"\n  First 3 rows preview:")
    print(df.head(3).to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────
# STEP 2: DATA CLEANING
# ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all data cleaning steps:
      1. Standardise column names
      2. Remove duplicate rows
      3. Check and handle missing values in score columns
      4. Validate score data types and ranges
      5. Standardise categorical values
    """
    print(f"\n{'=' * 60}")
    print("  STEP 2: DATA CLEANING")
    print(f"{'=' * 60}")

    original_rows = len(df)

    # ── 1. Standardise column names ──────────────────────────
    # Lowercase, strip whitespace, replace spaces/slashes with _
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    print(f"\n  [1] Column names standardised:")
    print(f"      {list(df.columns)}")

    # ── 2. Duplicate rows ─────────────────────────────────────
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"\n  [2] Duplicate rows found and removed: {duplicates}")
    print(f"      Rows after dedup: {len(df)}")

    # ── 3. Missing values ─────────────────────────────────────
    score_cols = ["math_score", "reading_score", "writing_score"]
    missing = df[score_cols].isna().sum()
    print(f"\n  [3] Missing values in score columns:")
    for col in score_cols:
        print(f"      {col}: {missing[col]} missing")

    rows_before = len(df)
    df = df.dropna(subset=score_cols)
    dropped = rows_before - len(df)
    print(f"      Rows dropped due to missing scores: {dropped}")

    # ── 4. Data type validation and score range check ─────────
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=score_cols)
    df = df[(df[score_cols] >= 0).all(axis=1) & (df[score_cols] <= 100).all(axis=1)]

    print(f"\n  [4] Score data types validated (numeric, 0-100):")
    for col in score_cols:
        print(f"      {col}: min={int(df[col].min())}, max={int(df[col].max())}, "
              f"dtype={df[col].dtype}")

    # ── 5. Standardise categorical values ────────────────────
    df["gender"] = df["gender"].str.strip().str.lower()
    df["race_ethnicity"] = df["race_ethnicity"].str.strip().str.title()
    df["parental_level_of_education"] = (
        df["parental_level_of_education"]
        .str.strip()
        .str.title()
        .str.replace(r"'S\b", "'s", regex=True)   # fix "Master'S" → "Master's"
    )
    df["lunch"] = df["lunch"].str.strip().str.lower()
    df["test_preparation_course"] = df["test_preparation_course"].str.strip().str.lower()

    print(f"\n  [5] Categorical values standardised:")
    print(f"      gender            : {sorted(df['gender'].unique().tolist())}")
    print(f"      race_ethnicity    : {sorted(df['race_ethnicity'].unique().tolist())}")
    print(f"      parental_edu      : {sorted(df['parental_level_of_education'].unique().tolist())}")
    print(f"      lunch             : {sorted(df['lunch'].unique().tolist())}")
    print(f"      test_prep         : {sorted(df['test_preparation_course'].unique().tolist())}")

    df = df.reset_index(drop=True)
    df["student_id"] = df.index + 1

    print(f"\n  Cleaning summary: {original_rows} rows in -> {len(df)} rows out")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING (Derived Columns)
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived columns:
      - total_score        = math + reading + writing
      - average_score      = total_score / 3  (rounded to 2dp)
      - performance_category: Excellent / Good / Average / Needs Improvement
      - pass_status        : Pass / Not Passing  (threshold = PASS_THRESHOLD)
    """
    print(f"\n{'=' * 60}")
    print("  STEP 3: FEATURE ENGINEERING")
    print(f"{'=' * 60}")

    # Total score
    df["total_score"] = df["math_score"] + df["reading_score"] + df["writing_score"]

    # Average score (rounded to 2 decimal places)
    df["average_score"] = (df["total_score"] / 3).round(2)

    # Performance category
    def assign_category(avg: float) -> str:
        if avg >= 90:
            return "Excellent"
        elif avg >= 75:
            return "Good"
        elif avg >= 60:
            return "Average"
        else:
            return "Needs Improvement"

    df["performance_category"] = df["average_score"].apply(assign_category)

    # Pass status
    df["pass_status"] = df["average_score"].apply(
        lambda x: "Pass" if x >= PASS_THRESHOLD else "Not Passing"
    )

    print(f"\n  New columns added:")
    print(f"  {'Column':<25} {'Description'}")
    print(f"  {DIVIDER}")
    print(f"  {'total_score':<25} math + reading + writing")
    print(f"  {'average_score':<25} total_score / 3  (rounded to 2dp)")
    print(f"  {'performance_category':<25} Excellent/Good/Average/Needs Improvement")
    print(f"  {'pass_status':<25} Pass if avg >= {PASS_THRESHOLD}, else Not Passing")

    print(f"\n  Performance category thresholds (analytical definitions):")
    for cat, (lo, hi) in PERFORMANCE_THRESHOLDS.items():
        print(f"    {cat:<25} avg {lo} – {hi}")

    print(f"\n  Sample rows (first 5):")
    sample_cols = ["student_id", "math_score", "reading_score", "writing_score",
                   "total_score", "average_score", "performance_category", "pass_status"]
    print(df[sample_cols].head(5).to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────
# STEP 4: DESCRIPTIVE STATISTICS & KPIs
# ─────────────────────────────────────────────────────────────

def descriptive_statistics(df: pd.DataFrame) -> None:
    """Compute and print key descriptive statistics and KPIs."""
    print(f"\n{'=' * 60}")
    print("  STEP 4: DESCRIPTIVE STATISTICS & KPIs")
    print(f"{'=' * 60}")

    total = len(df)
    avg_math    = round(df["math_score"].mean(), 2)
    avg_reading = round(df["reading_score"].mean(), 2)
    avg_writing = round(df["writing_score"].mean(), 2)
    overall_avg = round(df["average_score"].mean(), 2)
    pass_rate   = round((df["pass_status"] == "Pass").sum() / total * 100, 2)

    subject_avgs  = {"Math": avg_math, "Reading": avg_reading, "Writing": avg_writing}
    highest_subj  = max(subject_avgs, key=subject_avgs.get)
    lowest_subj   = min(subject_avgs, key=subject_avgs.get)

    print(f"\n  KPI CARDS")
    print(f"  {DIVIDER}")
    print(f"  {'Total Students':<30} {total}")
    print(f"  {'Average Math Score':<30} {avg_math}")
    print(f"  {'Average Reading Score':<30} {avg_reading}")
    print(f"  {'Average Writing Score':<30} {avg_writing}")
    print(f"  {'Overall Average Score':<30} {overall_avg}")
    print(f"  {'Pass Rate':<30} {pass_rate}%  (avg >= {PASS_THRESHOLD})")
    print(f"  {'Highest Average Subject':<30} {highest_subj}")
    print(f"  {'Lowest Average Subject':<30} {lowest_subj}")

    print(f"\n  FULL DESCRIPTIVE STATISTICS")
    print(f"  {DIVIDER}")
    score_cols = ["math_score", "reading_score", "writing_score", "total_score", "average_score"]
    print(df[score_cols].describe().round(2).to_string())

    print(f"\n  PERFORMANCE CATEGORY DISTRIBUTION")
    print(f"  {DIVIDER}")
    cat_order = ["Excellent", "Good", "Average", "Needs Improvement"]
    cat_counts = df["performance_category"].value_counts()
    for cat in cat_order:
        cnt = cat_counts.get(cat, 0)
        pct = round(cnt / total * 100, 1)
        bar = "#" * int(pct / 2)
        print(f"  {cat:<25} {cnt:>4} students  ({pct:>5.1f}%)  {bar}")

    print(f"\n  PASS / NOT PASSING BREAKDOWN")
    print(f"  {DIVIDER}")
    for status, cnt in df["pass_status"].value_counts().items():
        pct = round(cnt / total * 100, 1)
        print(f"  {status:<25} {cnt:>4} students  ({pct:>5.1f}%)")


# ─────────────────────────────────────────────────────────────
# STEP 5: SUBJECT PERFORMANCE ANALYSIS
# ─────────────────────────────────────────────────────────────

def subject_analysis(df: pd.DataFrame) -> None:
    """Analyse and compare average performance across subjects."""
    print(f"\n{'=' * 60}")
    print("  STEP 5: SUBJECT PERFORMANCE ANALYSIS")
    print(f"{'=' * 60}")

    subjects = {
        "Math":    df["math_score"],
        "Reading": df["reading_score"],
        "Writing": df["writing_score"],
    }

    print(f"\n  {'Subject':<12} {'Mean':>8} {'Median':>8} {'Std Dev':>8} {'Min':>6} {'Max':>6}")
    print(f"  {DIVIDER}")
    for subj, series in subjects.items():
        print(f"  {subj:<12} {series.mean():>8.2f} {series.median():>8.1f} "
              f"{series.std():>8.2f} {series.min():>6} {series.max():>6}")

    print(f"\n  Score Distribution (count of students per score band):")
    print(f"  {'Band':<12}", end="")
    for subj in subjects:
        print(f"  {subj:>10}", end="")
    print()
    print(f"  {DIVIDER}")
    bins = range(0, 101, 10)
    for i in range(10):
        lo = i * 10
        hi = lo + 10
        label = f"{lo}-{hi}"
        print(f"  {label:<12}", end="")
        for series in subjects.values():
            cnt = int(((series >= lo) & (series < hi)).sum())
            # Include 100 in the last bin
            if hi == 100:
                cnt = int(((series >= lo) & (series <= 100)).sum())
            print(f"  {cnt:>10}", end="")
        print()


# ─────────────────────────────────────────────────────────────
# STEP 6: DEMOGRAPHIC ANALYSIS
# ─────────────────────────────────────────────────────────────

def demographic_analysis(df: pd.DataFrame) -> None:
    """
    Compare average scores across demographic groups.

    IMPORTANT: These are observational comparisons only.
    Demographic characteristics do not cause differences in performance.
    Average scores varied across groups in this dataset.
    """
    print(f"\n{'=' * 60}")
    print("  STEP 6: DEMOGRAPHIC ANALYSIS")
    print(f"  NOTE: Observational comparisons only. No causal claims.")
    print(f"{'=' * 60}")

    groups = [
        ("Gender",                  "gender"),
        ("Race / Ethnicity",        "race_ethnicity"),
        ("Parental Education",      "parental_level_of_education"),
        ("Lunch Type",              "lunch"),
    ]

    for section_title, col in groups:
        print(f"\n  [{section_title}]")
        print(f"  {'Group':<35} {'Count':>6} {'Math':>7} {'Reading':>9} {'Writing':>9} {'Overall':>9}")
        print(f"  {DIVIDER}")

        grp = df.groupby(col).agg(
            count=("average_score", "count"),
            math_avg=("math_score", "mean"),
            reading_avg=("reading_score", "mean"),
            writing_avg=("writing_score", "mean"),
            overall_avg=("average_score", "mean"),
        ).sort_values("overall_avg", ascending=False).round(2)

        for name, row in grp.iterrows():
            print(f"  {str(name):<35} {int(row['count']):>6} {row['math_avg']:>7.2f} "
                  f"{row['reading_avg']:>9.2f} {row['writing_avg']:>9.2f} "
                  f"{row['overall_avg']:>9.2f}")


# ─────────────────────────────────────────────────────────────
# STEP 7: TEST PREPARATION ANALYSIS
# ─────────────────────────────────────────────────────────────

def test_prep_analysis(df: pd.DataFrame) -> None:
    """
    Compare students who completed vs did not complete the test
    preparation course.

    Note: This is an association observed in the dataset.
    It does not establish that completing the course causes
    higher scores (selection bias may be present).
    """
    print(f"\n{'=' * 60}")
    print("  STEP 7: TEST PREPARATION ANALYSIS")
    print(f"  NOTE: Association observed in dataset — not causal.")
    print(f"{'=' * 60}")

    grp = df.groupby("test_preparation_course").agg(
        count=("average_score", "count"),
        math_avg=("math_score", "mean"),
        reading_avg=("reading_score", "mean"),
        writing_avg=("writing_score", "mean"),
        overall_avg=("average_score", "mean"),
        pass_rate=("pass_status", lambda x: round((x == "Pass").sum() / len(x) * 100, 2)),
    ).round(2)

    total = len(df)
    print(f"\n  {'Metric':<25}", end="")
    for name in grp.index:
        label = "Completed" if name == "completed" else "Not Completed"
        print(f"  {label:>15}", end="")
    print()
    print(f"  {DIVIDER}")

    metrics = [
        ("count",       "Students"),
        ("math_avg",    "Avg Math"),
        ("reading_avg", "Avg Reading"),
        ("writing_avg", "Avg Writing"),
        ("overall_avg", "Overall Avg"),
        ("pass_rate",   "Pass Rate (%)"),
    ]

    for col, label in metrics:
        print(f"  {label:<25}", end="")
        for name in grp.index:
            val = grp.loc[name, col]
            if col == "count":
                pct = round(val / total * 100, 1)
                print(f"  {int(val):>10} ({pct}%)", end="")
            else:
                print(f"  {val:>15}", end="")
        print()

    # Dynamic insight
    if "completed" in grp.index and "none" in grp.index:
        diff = round(grp.loc["completed", "overall_avg"] - grp.loc["none", "overall_avg"], 2)
        direction = "higher" if diff > 0 else "lower"
        print(f"\n  Insight: In this dataset, students who completed the test preparation")
        print(f"  course had a {direction} average overall score by {abs(diff)} points.")
        print(f"  (Completed: {grp.loc['completed','overall_avg']} vs "
              f"Not Completed: {grp.loc['none','overall_avg']})")


# ─────────────────────────────────────────────────────────────
# STEP 8: SUBJECT CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> None:
    """
    Calculate Pearson correlation coefficients between subjects.

    Correlation measures the strength and direction of a linear
    association between two variables. A high correlation does
    NOT imply that one subject causes performance in another.
    """
    print(f"\n{'=' * 60}")
    print("  STEP 8: SUBJECT CORRELATION ANALYSIS")
    print(f"  NOTE: Correlation measures association, not causation.")
    print(f"{'=' * 60}")

    score_cols = ["math_score", "reading_score", "writing_score"]
    corr = df[score_cols].corr().round(4)

    print(f"\n  Pearson Correlation Matrix:")
    print(f"  {DIVIDER}")
    print(corr.rename(columns={"math_score": "Math", "reading_score": "Reading",
                                "writing_score": "Writing"})
            .rename(index={"math_score": "Math", "reading_score": "Reading",
                           "writing_score": "Writing"}).to_string())

    pairs = [
        ("Math vs Reading",    corr.loc["math_score",    "reading_score"]),
        ("Math vs Writing",    corr.loc["math_score",    "writing_score"]),
        ("Reading vs Writing", corr.loc["reading_score", "writing_score"]),
    ]

    print(f"\n  Pairwise Summary:")
    print(f"  {'Pair':<25} {'r':>8}  {'Strength'}")
    print(f"  {DIVIDER}")

    def strength(r: float) -> str:
        v = abs(r)
        if v >= 0.85: return "Very Strong positive association"
        if v >= 0.70: return "Strong positive association"
        if v >= 0.50: return "Moderate positive association"
        return "Weak association"

    for pair, r in pairs:
        print(f"  {pair:<25} {r:>8.4f}  {strength(r)}")

    strongest_pair = max(pairs, key=lambda x: x[1])
    print(f"\n  Strongest association: {strongest_pair[0]} (r = {strongest_pair[1]:.4f})")


# ─────────────────────────────────────────────────────────────
# STEP 9: KEY INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame) -> None:
    """
    Print a summary of key findings derived from the dataset.
    All values are calculated dynamically — nothing is hardcoded.
    Language is observational: 'higher average score in this dataset'
    rather than causal claims.
    """
    print(f"\n{'=' * 60}")
    print("  STEP 9: KEY INSIGHTS SUMMARY")
    print(f"{'=' * 60}")

    total = len(df)

    # Subject averages
    subj = {
        "Math":    df["math_score"].mean(),
        "Reading": df["reading_score"].mean(),
        "Writing": df["writing_score"].mean(),
    }
    best_subj  = max(subj, key=subj.get)
    worst_subj = min(subj, key=subj.get)

    # Test prep
    prep = df.groupby("test_preparation_course")["average_score"].mean()
    prep_diff = round(prep.get("completed", 0) - prep.get("none", 0), 2)

    # Parental education
    edu_avg = df.groupby("parental_level_of_education")["average_score"].mean()
    best_edu = edu_avg.idxmax()

    # Lunch
    lunch_avg = df.groupby("lunch")["average_score"].mean()
    std_avg = round(lunch_avg.get("standard", 0), 2)
    red_avg = round(lunch_avg.get("free/reduced", 0), 2)

    # Pass rate
    pass_rate = round((df["pass_status"] == "Pass").sum() / total * 100, 1)

    # Performance categories
    cat_counts = df["performance_category"].value_counts()

    # Strongest correlation
    cols = ["math_score", "reading_score", "writing_score"]
    corr = df[cols].corr()
    pairs = {
        "Math & Reading":    corr.loc["math_score",    "reading_score"],
        "Math & Writing":    corr.loc["math_score",    "writing_score"],
        "Reading & Writing": corr.loc["reading_score", "writing_score"],
    }
    strongest = max(pairs, key=pairs.get)

    insights = [
        (f"Highest average subject",
         f"{best_subj} (avg {round(subj[best_subj], 2)})"),

        (f"Lowest average subject",
         f"{worst_subj} (avg {round(subj[worst_subj], 2)})"),

        (f"Test preparation",
         f"Students who completed test prep had a {'higher' if prep_diff > 0 else 'lower'} "
         f"average score by {abs(prep_diff)} points in this dataset "
         f"({round(prep.get('completed',0),2)} vs {round(prep.get('none',0),2)})"),

        (f"Highest parental edu group",
         f"{best_edu} (avg {round(edu_avg[best_edu], 2)}) in this dataset"),

        (f"Lunch type comparison",
         f"Standard avg {std_avg} vs Free/Reduced avg {red_avg} in this dataset"),

        (f"Overall pass rate",
         f"{pass_rate}% of students achieved avg >= {PASS_THRESHOLD}"),

        (f"Excellent (avg >= 90)",
         f"{cat_counts.get('Excellent', 0)} students "
         f"({round(cat_counts.get('Excellent',0)/total*100,1)}%)"),

        (f"Good (avg 75–89)",
         f"{cat_counts.get('Good', 0)} students "
         f"({round(cat_counts.get('Good',0)/total*100,1)}%)"),

        (f"Average (avg 60–74)",
         f"{cat_counts.get('Average', 0)} students "
         f"({round(cat_counts.get('Average',0)/total*100,1)}%)"),

        (f"Needs Improvement (avg <60)",
         f"{cat_counts.get('Needs Improvement', 0)} students "
         f"({round(cat_counts.get('Needs Improvement',0)/total*100,1)}%)"),

        (f"Strongest subject correlation",
         f"{strongest} (r = {round(pairs[strongest], 4)}) — association, not causation"),
    ]

    print()
    for title, detail in insights:
        print(f"  >> {title}")
        print(f"     {detail}")
        print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  STUDENT PERFORMANCE ANALYTICS DASHBOARD")
    print("  Standalone Data Analysis Pipeline")
    print("=" * 60)

    # Load
    df = load_data(CSV_PATH)

    # Clean
    df = clean_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Analysis
    descriptive_statistics(df)
    subject_analysis(df)
    demographic_analysis(df)
    test_prep_analysis(df)
    correlation_analysis(df)
    generate_insights(df)

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print(f"  {len(df)} student records processed.")
    print("  Run  python backend/app.py  to launch the dashboard.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
