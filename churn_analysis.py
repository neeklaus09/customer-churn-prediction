"""
Customer Churn Prediction - Extended ML Project
--------------------------------------------------
An end-to-end churn prediction pipeline on the IBM Telco Customer Churn dataset.

Pipeline:
  1. Data loading & cleaning
  2. Exploratory Data Analysis (EDA) with saved plots
  3. Feature engineering & encoding
  4. Class imbalance handling (SMOTE)
  5. Model training: Logistic Regression, Random Forest, XGBoost
  6. Hyperparameter tuning (GridSearchCV) on the best-performing model
  7. Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  8. Feature importance visualization

Dataset: IBM Telco Customer Churn (7,032 customers, 19 features)
Author: <Your Name>
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_PATH = "telco.csv"
PLOTS_DIR = "plots"


# ---------------------------------------------------------------------------
# 1. Data loading & cleaning
# ---------------------------------------------------------------------------
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, out_dir: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Churn distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.savefig(f"{out_dir}/churn_distribution.png", bbox_inches="tight")
    plt.close()

    # Tenure vs churn
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y="tenure", data=df)
    plt.title("Tenure vs Churn")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.savefig(f"{out_dir}/tenure_vs_churn.png", bbox_inches="tight")
    plt.close()

    # Monthly charges vs churn
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges vs Churn")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.savefig(f"{out_dir}/monthlycharges_vs_churn.png", bbox_inches="tight")
    plt.close()

    print(f"EDA plots saved to '{out_dir}/'")


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df


# ---------------------------------------------------------------------------
# 5. Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }, confusion_matrix(y_test, preds)


def plot_confusion_matrix(cm, title, out_path):
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names, out_path, top_n=10):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_and_clean_data(DATA_PATH)
    run_eda(df, PLOTS_DIR)
    df = encode_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset size: {len(df)} customers, {len(X.columns)} features")
    print(f"Original churn rate: {y.mean() * 100:.1f}%")

    # --- Handle class imbalance with SMOTE (train set only) ---
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE, train set churn rate: {y_train_res.mean() * 100:.1f}%\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    all_results = {}

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train_res)
    lr_metrics, lr_cm = evaluate_model(lr, X_test_scaled, y_test)
    all_results["Logistic Regression"] = lr_metrics
    plot_confusion_matrix(lr_cm, "Logistic Regression - Confusion Matrix",
                           f"{PLOTS_DIR}/cm_logistic_regression.png")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train_res, y_train_res)
    rf_metrics, rf_cm = evaluate_model(rf, X_test, y_test)
    all_results["Random Forest"] = rf_metrics
    plot_confusion_matrix(rf_cm, "Random Forest - Confusion Matrix",
                           f"{PLOTS_DIR}/cm_random_forest.png")
    plot_feature_importance(rf, X.columns, f"{PLOTS_DIR}/feature_importance_rf.png")

    # --- XGBoost with light hyperparameter tuning ---
    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        xgb_params,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )
    xgb_grid.fit(X_train_res, y_train_res)
    best_xgb = xgb_grid.best_estimator_
    xgb_metrics, xgb_cm = evaluate_model(best_xgb, X_test, y_test)
    all_results["XGBoost (tuned)"] = xgb_metrics
    plot_confusion_matrix(xgb_cm, "XGBoost - Confusion Matrix",
                           f"{PLOTS_DIR}/cm_xgboost.png")
    plot_feature_importance(best_xgb, X.columns, f"{PLOTS_DIR}/feature_importance_xgb.png")

    print(f"Best XGBoost params: {xgb_grid.best_params_}\n")

    # --- Print all results ---
    for model_name, metrics in all_results.items():
        print(f"--- {model_name} ---")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value * 100:.2f}%")
        print()

    print(f"All plots saved to '{PLOTS_DIR}/' directory.")


if __name__ == "__main__":
    main()
