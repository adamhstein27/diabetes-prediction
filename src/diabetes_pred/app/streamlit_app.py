import streamlit as st
import textwrap
from diabetes_pred.inference.predict import predict

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ---------- Layout ----------
left_col, right_col = st.columns([1, 2], gap="large")

# ---------- Left Panel ----------
with left_col:
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown(
        textwrap.dedent(
            """
        This application estimates your risk of having diabetes based on various health metrics.
        
        **Instructions:**
        1. Fill in your health information in the right panel.
        2. Click the "Predict Risk" button to see your estimated risk of diabetes.
        
        **Note:** This tool is for informational purposes only and should not replace professional medical advice.
        """
        )
    )


# ---------- Right Panel ----------
with right_col:
    st.subheader("Enter your information")

    with st.form("prediction_form"):
        age = st.number_input("Age", 18, 100, 40)
        bmi = st.number_input("BMI", 10.0, 40.0, 25.0)

        col1, col2 = st.columns(2)
        with col1:
            ldl_cholesterol = st.number_input("LDL Cholesterol (mg/dL)", 50, 210, 100)
            triglycerides = st.number_input("Triglycerides (mg/dL)", 20, 300, 125)
        with col2:
            hdl_cholesterol = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 55)
            blood_pressure = st.number_input("Systolic BP (mm Hg)", 90, 200, 120)

        family_history = st.selectbox(
            "Family History of Diabetes",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        physical_activity = st.number_input(
            "Physical Activity (minutes per week)", 1, 1000, 150
        )

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        features = {
            "age": age,
            "bmi": bmi,
            "ldl_cholesterol": ldl_cholesterol,
            "hdl_cholesterol": hdl_cholesterol,
            "triglycerides": triglycerides,
            "systolic_bp": blood_pressure,
            "family_history_diabetes": family_history,
            "physical_activity_minutes_per_week": physical_activity,
        }

        prob = predict(features)

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Estimated Diabetes Risk",
            value=f"{prob:.2%}",
        )

        if prob >= 0.5:
            st.error(
                "⚠️ High estimated risk of diabetes. Please consult a healthcare professional."
            )
        else:
            st.success(
                "✅ Low estimated risk of diabetes. Maintain a healthy lifestyle!"
            )
