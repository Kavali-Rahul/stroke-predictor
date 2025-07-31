import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and related files
model = joblib.load("saved_model/stroke_rf_model.pkl")
feature_columns = joblib.load("saved_model/feature_columns.pkl")
with open("saved_model/best_threshold.txt", "r") as f:
    threshold = float(f.read())

# Load original dataset for visualization
df = pd.read_csv("stroke_data.csv")

# Streamlit UI
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")
st.title("🩺 Stroke Risk Prediction App")

st.markdown("Fill in the following details to predict your stroke risk:")

# Sidebar inputs
st.sidebar.header("Patient Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.slider("Age", 1, 100, 30)
hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Encode input
input_dict = {
    "gender": gender,
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,
}
input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)

# Ensure all columns match
for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_columns]

# Predict button
if st.button("🔍 Predict Stroke Risk"):
    prob = model.predict_proba(input_encoded)[0][1]
    prediction = int(prob >= threshold)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠ High risk of stroke! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of stroke. (Probability: {prob:.2f})")

    # Visualizations
    st.markdown("### Your Health vs Population Distribution")

    numeric_features = ["age", "avg_glucose_level", "bmi"]
    for feature in numeric_features:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=feature, fill=True, ax=ax, label="Population", color="skyblue")
        ax.axvline(input_dict[feature], color='orange', linestyle='--', linewidth=2, label="You")
        ax.set_title(f"{feature.replace('_',' ').title()} Distribution")
        ax.legend()
        st.pyplot(fig)

    # Health Overview
    st.markdown("### 🧠 Patient Health Overview")

    age_msg = "🟢 Your age is in the average range."
    if age > df["age"].mean() + 10:
        age_msg = "🔴 Your age is higher than most people."

    bmi_msg = "🟢 Your BMI is in the average range."
    if bmi > df["bmi"].mean() + 5:
        bmi_msg = "🔴 Your BMI is higher than average."

    glucose_msg = "🟢 Your glucose level is in the average range."
    if avg_glucose_level > df["avg_glucose_level"].mean() + 20:
        glucose_msg = "🔴 Your glucose level is high."

    st.markdown(age_msg)
    st.markdown(bmi_msg)
    st.markdown(glucose_msg)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by Rahul and Rakesh")