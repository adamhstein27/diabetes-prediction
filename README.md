# 🩺 Diabetes Risk Predictor

An end-to-end machine learning application that predicts an individual’s risk of diabetes using clinical and lifestyle features.  
The model is trained with **XGBoost** and served via a **Streamlit** web application.

This project demonstrates the full ML lifecycle: data processing, model training, evaluation, and interactive deployment.

---

## 🚀 Demo
👉 Live App: [Live Site](https://diabetes-prediction-27.streamlit.app/)

---

## 🧠 Model Overview

- **Algorithm:** XGBoost (classification)
- **Target:** Diabetes (binary)
- **Features include:**
  - Age
  - BMI
  - LDL cholesterol
  - HDL cholesterol
  - Triglycerides
  - Systolic blood pressure
  - Family history of diabetes
  - Physical activity level

The model is trained as a **scikit-learn pipeline** with feature scaling and persisted using `joblib`.

---

## 📊 Model Performance

Evaluated on a held-out test set:

| Metric     | Value |
|------------|-------|
| ROC AUC    | **0.72** |
| Accuracy   | **0.68** |
| Precision  | **0.70** |
| Recall     | **0.85** |
| F1 Score   | **0.77** |

> Metrics are computed using `evaluate.py` with a fixed random seed to ensure reproducibility.

---

## 📁 Project Structure

```text
.
├── data/
│   └── processed/
│       └── train.csv
├── models/
│   └── diabetes_xgb.joblib
├── src/
│   └── diabetes_pred/
│       ├── app/
│       │   └── streamlit_app.py
│       ├── inference/
│       │   └── predict.py
│       ├── modeling/
│       │   ├── model.py
│       │   ├── train.py
│       │   └── evaluate.py
│       └── data/
│           └── schema.py
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md

```
---

## 🛠️ Setup Instructions

1. **Clone the repository:**
```bash
   git clone https://github.com/adamhstein27/diabetes_pred.git
   cd diabetes_pred
```

2. **Install dependencies:**
```bash
   uv sync
```

3. **Create and activate a virtual environment:**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

4. **Train the model:**
```bash
   uv run python -m diabetes_pred.modeling.train
```

5. **Run the Streamlit app:**
```bash
   uv run streamlit run src/diabetes_pred.app.streamlit_app
```