"""
Predict pass/fail probability for one student using the trained model.

Example:
  python predict.py --attendance 85 --study-hours 10 --previous-grade 70 \
                    --assignment 72 --midterm 65

Any value you leave out is treated as missing and filled automatically.
"""

import argparse

import joblib
import pandas as pd

parser = argparse.ArgumentParser(description="Predict a student's pass/fail probability")
parser.add_argument("--attendance", type=float, help="attendance percentage (0-100)")
parser.add_argument("--study-hours", type=float, help="study hours per week")
parser.add_argument("--previous-grade", type=float, help="previous grade (0-100)")
parser.add_argument("--assignment", type=float, help="assignment average (0-100)")
parser.add_argument("--midterm", type=float, help="midterm score (0-100)")
args = parser.parse_args()

saved = joblib.load("model.joblib")
student = pd.DataFrame([{
    "attendance_pct": args.attendance,
    "study_hours_per_week": args.study_hours,
    "previous_grade": args.previous_grade,
    "assignment_avg": args.assignment,
    "midterm_score": args.midterm,
}])[saved["features"]].astype(float)

p_pass = saved["model"].predict_proba(student)[0, 1]
print(f"Model used: {saved['name']}")
print(f"Probability of passing: {p_pass:.1%}")
print(f"Probability of failing: {1 - p_pass:.1%}")
print("Prediction:", "PASS" if p_pass >= 0.5 else "FAIL")
