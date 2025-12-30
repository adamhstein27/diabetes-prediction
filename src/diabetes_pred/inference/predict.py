import joblib
import pandas as pd
import streamlit as st

from diabetes_pred.data.schema import FEATURE_NAMES

MODEL_PATH = "models/diabetes_xgb.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> float:
    model = load_model()

    df = pd.DataFrame([features])
    df = df[FEATURE_NAMES] 

    proba = model.predict_proba(df)[0, 1]
    return float(proba)
