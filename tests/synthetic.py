"""Fake Telco-style data for tests and CI. Never use it for a real deployment."""
import numpy as np
import pandas as pd

COLS = {
    "gender": ["Male", "Female"], "Partner": ["Yes", "No"], "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"], "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}


def make_dataset(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in COLS.items()})
    df.insert(0, "customerID", [f"C{i:05d}" for i in range(n)])
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 72, n)
    df["MonthlyCharges"] = rng.uniform(20, 120, n).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # real Telco data has blank strings here
    df["numAdminTickets"] = rng.integers(0, 4, n)
    df["numTechTickets"] = rng.integers(0, 4, n)
    # churn more likely: month-to-month, short tenure, high bill
    z = (-1.0 + 1.6 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"]
         + 0.02 * df["MonthlyCharges"])
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    return df


SAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5,
    "TotalCharges": 447.5, "numAdminTickets": 1, "numTechTickets": 2,
}
