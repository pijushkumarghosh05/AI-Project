"""
Streamlit web app for the Student Performance Prediction project.
Loads model.joblib (created by student_performance.py) and shows
pass/fail probabilities for one student.

Run locally:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


saved = load_model()
model, features = saved["model"], saved["features"]

st.title("Student Performance Prediction")
st.write("Enter a student's details to predict the probability of **passing or failing**.")

with st.form("student_form"):
    st.subheader("Student information")
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.number_input("Attendance (%)", 0.0, 100.0, 85.0, step=1.0)
        study_hours = st.number_input("Study hours per week", 0.0, 80.0, 10.0, step=0.5)
        previous_grade = st.number_input("Previous grade (0-100)", 0.0, 100.0, 70.0, step=1.0)
    with col2:
        assignment = st.number_input("Assignment average (0-100)", 0.0, 100.0, 72.0, step=1.0)
        midterm = st.number_input("Midterm score (0-100)", 0.0, 100.0, 65.0, step=1.0)
    submitted = st.form_submit_button("Predict Result", use_container_width=True)

if submitted:
    student = pd.DataFrame([{
        "attendance_pct": attendance,
        "study_hours_per_week": study_hours,
        "previous_grade": previous_grade,
        "assignment_avg": assignment,
        "midterm_score": midterm,
    }])[features].astype(float)

    p_pass = float(model.predict_proba(student)[0, 1])

    if p_pass >= 0.5:
        st.success(f"Prediction: PASS ({p_pass:.1%} chance of passing)")
    else:
        st.error(f"Prediction: FAIL ({1 - p_pass:.1%} chance of failing)")

    c1, c2 = st.columns(2)
    c1.metric("Probability of passing", f"{p_pass:.1%}")
    c2.metric("Probability of failing", f"{1 - p_pass:.1%}")
    st.progress(p_pass)
    st.caption(f"Model used: {saved['name']}. Predictions are probabilities meant to "
               "support students, not final judgements about them.")

results_path = BASE / "results.csv"
if results_path.exists():
    with st.expander("Decision Tree vs Random Forest results"):
        st.dataframe(pd.read_csv(results_path).set_index("Model"))
