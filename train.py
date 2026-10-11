"""Train the churn model and save a single sklearn Pipeline for the API.

Usage:
    python train.py --data Customer_Churn_Dataset.xlsx
    python train.py --synthetic          # demo/CI model only, NOT for real use

The saved Pipeline contains preprocessing + classifier, so the API can accept raw
customer fields (no manual one-hot encoding at inference time).
"""
import argparse
import json
import typing
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas import CustomerFeatures

FEATURES = list(CustomerFeatures.model_fields)
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "numAdminTickets", "numTechTickets"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning steps as the notebook."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop(columns=["customerID"], errors="ignore")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def check_schema(df: pd.DataFrame) -> None:
    """Fail loudly if the data has columns/values the API schema would reject."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    for col in CATEGORICAL:
        allowed = set(typing.get_args(CustomerFeatures.model_fields[col].annotation))
        unseen = set(df[col].dropna().unique()) - allowed
        if unseen:
            raise ValueError(f"Column '{col}' has values not in app/schemas.py: {sorted(unseen)}. "
                             "Update the Literal for that field.")


def build_pipeline(classifier) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("clf", classifier)])


def train(df: pd.DataFrame, out_dir: str, data_source: str = "real") -> dict:
    df = clean(df)
    check_schema(df)
    X, y = df[FEATURES], df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = build_pipeline(clf).fit(X_train, y_train)
        pred, proba = pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1]
        rep = classification_report(y_test, pred, output_dict=True)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "churn_precision": round(rep["1"]["precision"], 4),
            "churn_recall": round(rep["1"]["recall"], 4),
        }
        fitted[name] = pipe
        print(f"{name}: {results[name]}")

    best = max(results, key=lambda n: results[n]["roc_auc"])  # ROC-AUC suits imbalanced churn data
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], out / "churn_model.joblib", compress=3)
    metrics = {
        "selected_model": best,
        "data_source": data_source,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "results": results,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved '{best}' to {out / 'churn_model.joblib'}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="path to Customer_Churn_Dataset.xlsx or .csv")
    p.add_argument("--synthetic", action="store_true", help="train on fake data (demo/CI only)")
    p.add_argument("--out", default="model")
    a = p.parse_args()
    if a.synthetic:
        from tests.synthetic import make_dataset
        train(make_dataset(2000), a.out, data_source="synthetic")
    elif a.data:
        train(load_data(a.data), a.out)
    else:
        p.error("provide --data PATH (or --synthetic for a demo model)")


if __name__ == "__main__":
    main()
