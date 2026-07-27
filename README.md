Customer Churn Prediction — Extended ML Project

An end-to-end churn prediction pipeline on the IBM Telco Customer Churn dataset (7,032 customers, 19 features). Goes beyond a baseline model by adding EDA, class-imbalance handling, a third model (XGBoost), hyperparameter tuning, and visualizations.

Pipeline
Data cleaning — handle missing TotalCharges, encode target
EDA — churn distribution, tenure vs churn, monthly charges vs churn (saved as plots)
Feature engineering — label encoding of categorical variables
Class imbalance handling — SMOTE oversampling on the training set (raw churn rate 26.6% → balanced 50% for training)
Modeling — Logistic Regression, Random Forest, and XGBoost (tuned via GridSearchCV)
Evaluation — accuracy, precision, recall, F1, ROC-AUC, confusion matrices
Feature importance — visualized for Random Forest and XGBoost
Results (after SMOTE)
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	73.6%	50.3%	71.7%	59.1%	80.9%
Random Forest	75.8%	53.5%	69.3%	60.4%	82.0%
XGBoost (tuned)	75.9%	54.3%	59.1%	56.6%	80.9%

Best tuned XGBoost params: learning_rate=0.1, max_depth=5, n_estimators=200

Note: SMOTE improved recall significantly (RF recall rose from ~49% baseline to ~69%) at a small cost to precision/accuracy — a deliberate trade-off since catching more actual churners is usually more valuable than raw accuracy in a retention use case.

Top predictive features: tenure, contract type, total charges, monthly charges, online security.

Visualizations included (/plots)
Churn distribution
Tenure vs churn, Monthly charges vs churn
Confusion matrices (all 3 models)
Feature importance (Random Forest, XGBoost)
How to run
bash
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
python churn_extended.py

Place telco.csv (IBM Telco Customer Churn dataset) in the same folder — available on Kaggle.

Tech stack

Python · Pandas · Scikit-learn · XGBoost · Imbalanced-learn (SMOTE) · Matplotlib · Seaborn

Possible future work
SHAP values for per-prediction explainability
Deploy as a simple Flask/Streamlit app for live scoring
Cost-sensitive learning tied to actual retention-offer cost vs. customer lifetime value

- Handle class imbalance (SMOTE, class weights) to improve recall
- Hyperparameter tuning (GridSearchCV)
- Try XGBoost / LightGBM
- Add SHAP values for interpretability
