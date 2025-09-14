import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------- Load model and features ----------------------
model = joblib.load("saved_model/stroke_rf_model.pkl")
top_features = joblib.load("saved_model/feature_columns.pkl")
with open("saved_model/best_threshold.txt", "r") as f:
    threshold = float(f.read())

# Load original dataset for visualization
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
Residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
smoking_status = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# ---------------------- Encode input ----------------------
input_dict = {
    "age": age,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "hypertension": 1 if hypertension=="Yes" else 0,
    "heart_disease": 1 if heart_disease=="Yes" else 0,
}

# Add categorical one-hot encoding
for col in top_features:
    if col.startswith("gender_"):
        input_dict[col] = 1 if col=="gender_"+gender else 0
    elif col.startswith("ever_married_"):
        input_dict[col] = 1 if col=="ever_married_"+ever_married else 0
    elif col.startswith("work_type_"):
        input_dict[col] = 1 if col=="work_type_"+work_type else 0
    elif col.startswith("Residence_type_"):
        input_dict[col] = 1 if col=="Residence_type_"+Residence_type else 0
    elif col.startswith("smoking_status_"):
        input_dict[col] = 1 if col=="smoking_status_"+smoking_status else 0

input_df = pd.DataFrame([input_dict])
input_df = input_df[top_features]  # ensure correct column order

# ---------------------- Prediction ----------------------
if st.button("🔍 Predict Stroke Risk"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= threshold)

    st.subheader("Prediction Result")
    if prediction==1:
        st.error(f"⚠ High risk of stroke! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of stroke. (Probability: {prob:.2f})")

    # ---------------------- Visualizations ----------------------
    st.markdown("### Your Health vs Population Distribution")
    numeric_features = ["age", "bmi", "avg_glucose_level"]

    col1, col2 = st.columns(2)
    
    # First column: age & bmi
    with col1:
        for feature in ["age","bmi"]:
            fig, ax = plt.subplots()
            sns.kdeplot(df[feature], fill=True, ax=ax, color="skyblue", label="Population")
            ax.axvline(input_dict[feature], color="orange", linestyle="--", linewidth=2, label="You")
            ax.set_title(f"{feature.replace('_',' ').title()} Distribution")
            ax.legend()
            st.pyplot(fig)

    # Second column: glucose
    with col2:
        feature = "avg_glucose_level"
        fig, ax = plt.subplots()
        sns.kdeplot(df[feature], fill=True, ax=ax, color="skyblue", label="Population")
        ax.axvline(input_dict[feature], color="orange", linestyle="--", linewidth=2, label="You")
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
st.markdown("👨‍💻 Developed by Rahul")
