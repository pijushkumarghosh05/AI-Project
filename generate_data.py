"""
Creates a realistic but SYNTHETIC student dataset (students.csv) with the
usual problems of real academic data: missing grades, impossible values
and duplicate rows. Use it if you don't have your own dataset.

If you have a real CSV, skip this file and give your CSV to
student_performance.py instead (column names are listed in that file).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1000

# A hidden "ability" level makes the columns realistically related to each other
ability = rng.normal(0, 1, n)

study_hours = np.clip(12 + 3 * ability + rng.normal(0, 4, n), 0, 40)
attendance = np.clip(80 + 6 * ability + rng.normal(0, 9, n), 30, 100)
previous_grade = np.clip(65 + 12 * ability + rng.normal(0, 6, n), 20, 100)
assignment_avg = np.clip(68 + 9 * ability + 0.4 * (study_hours - 12) + rng.normal(0, 7, n), 0, 100)
midterm_score = np.clip(60 + 13 * ability + 0.5 * (study_hours - 12) + rng.normal(0, 8, n), 0, 100)

# The pass/fail label depends on all columns plus some luck
final_score = (0.35 * midterm_score + 0.20 * assignment_avg + 0.15 * previous_grade
               + 0.10 * attendance + 0.60 * study_hours + rng.normal(0, 6, n))

df = pd.DataFrame({
    "student_id": np.arange(1, n + 1),
    "attendance_pct": attendance.round(1),
    "study_hours_per_week": study_hours.round(1),
    "previous_grade": previous_grade.round(1),
    "assignment_avg": assignment_avg.round(1),
    "midterm_score": midterm_score.round(1),
    "result": np.where(final_score >= 55, "Pass", "Fail"),
})

# --- make the data messy on purpose ---
for col in ["previous_grade", "assignment_avg", "midterm_score"]:
    df.loc[rng.random(n) < 0.07, col] = np.nan      # missing grades
df.loc[rng.random(n) < 0.03, "attendance_pct"] = np.nan
df.loc[rng.choice(n, 8, replace=False), "attendance_pct"] = 120      # impossible
df.loc[rng.choice(n, 6, replace=False), "study_hours_per_week"] = -5  # impossible
df.loc[rng.choice(n, 5, replace=False), "midterm_score"] = 150        # impossible
df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)  # duplicates

df.to_csv("students.csv", index=False)
print(f"Saved students.csv with {len(df)} rows")
