# Customer Churn Prediction (ML)

Predicts customer churn using the **IBM Telco Customer Churn dataset** (7,032 customers, 19 features). Trains and compares Logistic Regression and Random Forest classifiers, and reports which features drive churn the most.

## Results

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 79.4%    | 62.4%     | 56.4%  | 59.3%  | 83.5%   |
| Random Forest       | 79.3%    | 64.4%     | 48.9%  | 55.6%  | 83.4%   |

**Top predictive features (Random Forest):** tenure, contract type, total charges, monthly charges, online security.

## How to run

```bash
pip install pandas scikit-learn
python churn_prediction.py
```

Make sure `telco.csv` (IBM Telco Customer Churn dataset) is in the same folder. You can get it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or the IBM public repo.

## Tech stack

Python, Pandas, Scikit-learn (Logistic Regression, Random Forest)

## Possible improvements

- Handle class imbalance (SMOTE, class weights) to improve recall
- Hyperparameter tuning (GridSearchCV)
- Try XGBoost / LightGBM
- Add SHAP values for interpretability
