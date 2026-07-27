"""
Customer Churn Prediction using Machine Learning
--------------------------------------------------
Predicts customer churn using the IBM Telco Customer Churn dataset.
Trains and evaluates Logistic Regression and Random Forest classifiers,
and reports the top features driving churn.

Dataset: IBM Telco Customer Churn (7,032 customers, 19 features)
Author: <Your Name>
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_PATH = "telco.csv"


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load the dataset and handle missing/incorrect types."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns."""
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df


def evaluate_model(model, X_test, y_test) -> dict:
    """Return standard classification metrics for a trained model."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    df = load_and_clean_data(DATA_PATH)
    df = encode_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset size: {len(df)} customers, {len(X.columns)} features")
    print(f"Churn rate: {y.mean() * 100:.1f}%")
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}\n")

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_metrics = evaluate_model(lr, X_test_scaled, y_test)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test)

    results = {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
    }

    for model_name, metrics in results.items():
        print(f"--- {model_name} ---")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value * 100:.2f}%")
        print()

    # --- Feature importance (Random Forest) ---
    importances = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    print("Top 5 predictive features (Random Forest):")
    print(importances.head(5))


if __name__ == "__main__":
    main()
