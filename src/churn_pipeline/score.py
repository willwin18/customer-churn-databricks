"""
score.py
Generates churn risk scores for all customers using a trained model.
"""

import pandas as pd


def generate_churn_scores(model, scaler, df_pandas: pd.DataFrame, features: list) -> pd.DataFrame:
    """Add churn_risk_score column to the full customer dataset."""
    X_all_scaled = scaler.transform(df_pandas[features])
    df_pandas["churn_risk_score"] = model.predict_proba(X_all_scaled)[:, 1]
    return df_pandas
