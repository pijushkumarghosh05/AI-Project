"""
Student Performance Prediction
==============================
Predicts whether a student will Pass or Fail from attendance, study hours
and academic records, and compares a Decision Tree with a Random Forest.

Steps
  1. Load and clean the data with Pandas
  2. Split into train / test sets
  3. Handle missing grades (median imputation, learned from training data only)
  4. Train Decision Tree and Random Forest (small hyper-parameter search)
  5. Compare accuracy and other metrics
  6. Save the best model to model.joblib (used by predict.py)

Expected CSV columns:
  attendance_pct, study_hours_per_week, previous_grade,
  assignment_avg, midterm_score, result (Pass/Fail)

Run:  python student_performance.py [path/to/data.csv]
"""

import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["attendance_pct", "study_hours_per_week", "previous_grade",
            "assignment_avg", "midterm_score"]
TARGET = "result"

# Values outside these ranges are impossible and treated as missing
VALID_RANGES = {
    "attendance_pct": (0, 100),
    "study_hours_per_week": (0, 80),
    "previous_grade": (0, 100),
    "assignment_avg": (0, 100),
    "midterm_score": (0, 100),
}

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. LOAD AND CLEAN
# ---------------------------------------------------------------------------
def load_and_clean(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows")

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    # Make sure numeric columns really are numbers (bad text becomes NaN)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impossible values (e.g. attendance 120%) -> NaN so they get imputed later
    for col, (low, high) in VALID_RANGES.items():
        invalid = ~df[col].between(low, high) & df[col].notna()
        if invalid.sum():
            print(f"Found {invalid.sum()} impossible values in '{col}' -> set to missing")
        df[col] = df[col].where(df[col].between(low, high))

    # Target: Pass -> 1, Fail -> 0. Rows without a result can't be used.
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower().map({"pass": 1, "fail": 0})
    before = len(df)
    df = df.dropna(subset=[TARGET])
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with no pass/fail result")
    df[TARGET] = df[TARGET].astype(int)

    print("\nMissing values per column after cleaning:")
    print(df[FEATURES].isna().sum().to_string())
    print(f"\nClass balance: {df[TARGET].mean():.0%} pass, {1 - df[TARGET].mean():.0%} fail\n")
    return df


# ---------------------------------------------------------------------------
# 2. MODELS
# ---------------------------------------------------------------------------
def make_search(model, param_grid):
    """Imputer + model in one Pipeline, tuned with 5-fold cross-validation.

    Putting the imputer inside the pipeline means the median used to fill
    missing grades is calculated only from training data in each fold, so
    no information leaks from the test set.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])
    return GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)


def evaluate(name, search, X_test, y_test):
    pred = search.predict(X_test)
    prob = search.predict_proba(X_test)[:, 1]
    return {
        "Model": name,
        "CV accuracy": search.best_score_,
        "Test accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC AUC": roc_auc_score(y_test, prob),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "students.csv"
    df = load_and_clean(path)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    searches = {
        "Decision Tree": make_search(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [3, 4, 5, 6, 8, None],
             "model__min_samples_leaf": [1, 5, 10]}),
        "Random Forest": make_search(
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 300],
             "model__max_depth": [None, 6, 10],
             "model__min_samples_leaf": [1, 5]}),
    }

    results = []
    for name, search in searches.items():
        print(f"Training {name}...")
        search.fit(X_train, y_train)
        print(f"  best settings: {search.best_params_}")
        results.append(evaluate(name, search, X_test, y_test))

    table = pd.DataFrame(results).set_index("Model").round(3)
    print("\n=== Decision Tree vs Random Forest ===")
    print(table.to_string())
    table.to_csv("results.csv")

    # Pick the winner by cross-validation score (not by the test set)
    best_name = max(searches, key=lambda k: searches[k].best_score_)
    best_model = searches[best_name].best_estimator_
    print(f"\nBest model: {best_name}")

    # Which inputs matter most?
    importances = pd.Series(best_model.named_steps["model"].feature_importances_,
                            index=FEATURES).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importances.round(3).to_string())

    joblib.dump({"model": best_model, "features": FEATURES, "name": best_name}, "model.joblib")
    print("\nSaved model to model.joblib  (use predict.py to get pass/fail probabilities)")


if __name__ == "__main__":
    main()
