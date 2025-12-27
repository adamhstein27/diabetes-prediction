import streamlit as st
from diabetes_pred.inference.predict import predict

st.set_page_config(page_title="Diabetes Risk Predictor")

st.title("🩺 Diabetes Risk Predictor")

age = st.number_input("Age", 18, 100, 40)
bmi = st.number_input("BMI", 10.0, 40.0, 25.0)
ldl_cholesterol = st.number_input("LDL Cholesterol (mg/dL)", 50, 210, 100)
hdl_cholesterol = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 55)
triglycerides = st.number_input("Triglycerides (mg/dL)", 20, 300, 125)
blood_pressure = st.number_input("Systolic Blood Pressure (mm Hg)", 90, 200, 120)
family_history = st.selectbox("Family History of Diabetes", options=[0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No")
physical_activity = st.number_input("Physical Activity (minutes per week)", 1, 1000, 150)

if st.button("Predict Risk"):
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
    st.metric("Diabetes Risk Probability", f"{prob:.2%}")
    if prob >= 0.5:
        st.error("High risk of diabetes. Please consult a healthcare professional.")
    else:
        st.success("Low risk of diabetes. Maintain a healthy lifestyle!")