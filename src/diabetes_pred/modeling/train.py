# src/diabetes_pred/modeling/train.py
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diabetes_pred.modeling.model import build_model
from diabetes_pred.data.schema import FEATURE_NAMES, TARGET_COLUMN

MODEL_PATH = Path("models/diabetes_xgb.joblib")


def train(df: pd.DataFrame):
    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", build_model()),
        ]
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/train.csv")
    train(df)
