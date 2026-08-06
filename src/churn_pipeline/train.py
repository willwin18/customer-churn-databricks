"""
train.py
Trains and evaluates churn prediction models.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


FEATURES = ["total_spend", "num_orders"]


def prepare_training_data(df_pandas: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data into train/test sets."""
    X = df_pandas[FEATURES]
    y = df_pandas["churned"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_logistic_regression(X_train, y_train):
    """Train a scaled logistic regression model. Returns (model, scaler)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate_model(model, scaler, X_test, y_test) -> dict:
    """Evaluate a trained model, returns a dict of metrics."""
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
