from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from diabetes_pred.data.schema import FEATURE_NAMES, TARGET_COLUMN

MODEL_PATH = Path("models/diabetes_xgb.joblib")


def evaluate():
    df = pd.read_csv("data/processed/train.csv")

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    # SAME split as training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = joblib.load(MODEL_PATH)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate()
