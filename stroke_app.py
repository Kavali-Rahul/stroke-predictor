import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------- Load model and preprocessing ----------------------
model = joblib.load("saved_model/best_stroke_model.pkl")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define which columns are categorical and numerical
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

# Rebuild preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols)
])

# Fit it on stroke_data.csv (same columns you used in training)
df = pd.read_csv("stroke_data.csv")
df["bmi"].fillna(df["bmi"].mean(), inplace=True)
X = df[categorical_cols + numeric_cols]
preprocessor.fit(X)
with open("saved_model/best_threshold.txt", "r") as f:
    threshold = float(f.read())

# Load dataset for visualization
df = pd.read_csv("stroke_data.csv")
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")
st.title("🩺 Stroke Risk Prediction App")
st.markdown("Fill in the following details to predict your stroke risk:")

# ---------------------- Sidebar inputs ----------------------
st.sidebar.header("Patient Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.slider("Age", 1, 100, 30)
hypertension = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# ---------------------- Prepare input DataFrame ----------------------
input_raw = pd.DataFrame([{
    "age": age,
    "hypertension": 1 if hypertension=="Yes" else 0,
    "heart_disease": 1 if heart_disease=="Yes" else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "gender": gender,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "smoking_status": smoking_status
}])

# ---------------------- Fill missing columns for preprocessor ----------------------
for col in preprocessor.feature_names_in_:
    if col not in input_raw.columns:
        input_raw[col] = 0  # Fill missing categorical columns with 0

input_raw = input_raw[preprocessor.feature_names_in_]  # Ensure correct column order

# ---------------------- Predict when button clicked ----------------------
if st.button("🔍 Predict Stroke Risk"):
    input_preprocessed = preprocessor.transform(input_raw)
    
    prob = model.predict_proba(input_preprocessed)[0][1]
    prediction = int(prob >= threshold)

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠ High risk of stroke! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of stroke. (Probability: {prob:.2f})")

    # ---------------------- Visualizations ----------------------
    st.markdown("### Your Health vs Population Distribution")
    numeric_features = ["age", "bmi", "avg_glucose_level"]
    col1, col2 = st.columns(2)

    with col1:
        for feature in ["age","bmi"]:
            fig, ax = plt.subplots()
            sns.kdeplot(df[feature], fill=True, ax=ax, color="skyblue", label="Population")
            ax.axvline(input_raw[feature][0], color="orange", linestyle="--", linewidth=2, label="You")
            ax.set_title(f"{feature.replace('_',' ').title()} Distribution")
            ax.legend()
            st.pyplot(fig)

    with col2:
        feature = "avg_glucose_level"
        fig, ax = plt.subplots()
        sns.kdeplot(df[feature], fill=True, ax=ax, color="skyblue", label="Population")
        ax.axvline(input_raw[feature][0], color="orange", linestyle="--", linewidth=2, label="You")
        ax.set_title(f"{feature.replace('_',' ').title()} Distribution")
        ax.legend()
        st.pyplot(fig)

    # ---------------------- Health Overview ----------------------
    st.markdown("### 🧠 Patient Health Overview")
    age_msg = "🟢 Your age is in the average range." if age <= df["age"].mean()+10 else "🔴 Your age is higher than most people."
    bmi_msg = "🟢 Your BMI is in the average range." if bmi <= df["bmi"].mean()+5 else "🔴 Your BMI is higher than average."
    glucose_msg = "🟢 Your glucose level is in the average range." if avg_glucose_level <= df["avg_glucose_level"].mean()+20 else "🔴 Your glucose level is high."
    st.markdown(age_msg)
    st.markdown(bmi_msg)
    st.markdown(glucose_msg)

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("👨‍💻 Developed by Rahul and Rakesh")