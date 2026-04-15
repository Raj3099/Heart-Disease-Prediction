import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Heart Disease Prediction App")
st.write("Enter the patient details below to predict the likelihood of heart disease.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)

with col2:
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Pre-processing the inputs to match the model's 15 features
def preprocess_input():
    # Mapping categorical inputs to match the model's expected feature names
    data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_M': 1 if sex == "Male" else 0,
        'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
        'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
        'ChestPainType_TA': 1 if chest_pain == "TA" else 0,
        'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
        'RestingECG_ST': 1 if resting_ecg == "ST" else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == "Yes" else 0,
        'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
        'ST_Slope_Up': 1 if st_slope == "Up" else 0
    }
    return pd.DataFrame([data])

input_df = preprocess_input()

if st.button("Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f"High Risk of Heart Disease (Probability: {probability[0][1]:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease (Probability: {probability[0][0]:.2f})")

st.info("Note: This is a machine learning demo and not a substitute for professional medical advice.")
