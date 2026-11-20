# Student Performance Prediction (Education Analytics)

Predicts whether a student will **pass or fail** from attendance, weekly study hours and academic records, and compares a **Decision Tree** with a **Random Forest**.

## Website Link

**Live app:** https://student-performance-pijush.streamlit.app/

Anyone can open the link, enter a student's details and see the pass/fail probabilities. No installation needed.

## Project files

| File | Purpose |
|------|---------|
| `generate_data.py` | Creates a synthetic, deliberately messy dataset (`students.csv`). Skip it if you use your own CSV. |
| `student_performance.py` | Cleans the data, trains and compares both models, saves the best one. |
| `predict.py` | Loads the saved model and prints pass/fail probabilities for one student. |
| `app.py` | Streamlit website: enter student details and get pass/fail probabilities in the browser. |
| `model.joblib` | The trained model used by `app.py` and `predict.py`. |
| `students.csv` | The synthetic dataset created by `generate_data.py`. |
| `results.csv` | Metrics of the last training run. |

## Pipeline

1. **Cleaning (Pandas):** remove duplicate rows, convert columns to numbers, mark impossible values (e.g. attendance 120%, negative study hours) as missing, drop rows with no result.
2. **Missing grades:** filled with the column **median** using scikit-learn's `SimpleImputer`. It sits inside a `Pipeline`, so the median is calculated from training data only (no data leakage).
3. **Split:** 80% train / 20% test, stratified so both sets keep the same pass/fail ratio.
4. **Models:** Decision Tree and Random Forest, each tuned with `GridSearchCV` (5-fold cross-validation).
5. **Comparison:** accuracy, precision, recall, F1 and ROC AUC; the best model is chosen by cross-validation score and saved to `model.joblib`.

## Run

```bash
pip install -r requirements.txt
python generate_data.py            # only if you don't have your own data
python student_performance.py      # or: python student_performance.py my_data.csv
streamlit run app.py               # open the website locally
python predict.py --attendance 85 --study-hours 10 --previous-grade 70 --assignment 72 --midterm 65
```

Your own CSV needs these columns: `attendance_pct, study_hours_per_week, previous_grade, assignment_avg, midterm_score, result` (Pass/Fail).

## Results (synthetic data, test set of 200 students)

| Model | CV accuracy | Test accuracy | F1 | ROC AUC |
|-------|-------------|---------------|-----|---------|
| Decision Tree | 0.819 | 0.875 | 0.899 | 0.927 |
| Random Forest | 0.851 | 0.870 | 0.896 | 0.931 |

The Random Forest is more stable (higher cross-validation accuracy and ROC AUC). On this small test set the two are within noise of each other. Most important features: midterm score, previous grade, assignment average.

## Limitations

- The bundled data is synthetic, so real-world accuracy will differ.
- Predictions are probabilities to support students, not final judgements about them.
