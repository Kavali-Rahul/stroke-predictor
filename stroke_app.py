import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and other saved files
model = joblib.load("saved_model/stroke_rf_model.pkl")
feature_columns = joblib.load("saved_model/feature_columns.pkl")
with open("saved_model/best_threshold.txt", "r") as f:
    best_threshold = float(f.read())

# Load main dataset for population comparison
df = pd.read_csv("stroke_data.csv")

# Title
st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")
st.title("🧠 Stroke Risk Prediction Web App")
st.write("Predict stroke risk and compare your health with the population.")

# Sidebar input
st.sidebar.header("Enter Patient Information")

def user_input():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female", "Other"))
    age = st.sidebar.slider("Age", 1, 100, 30)
    hypertension = st.sidebar.selectbox("Hypertension", ("No", "Yes"))
    heart_disease = st.sidebar.selectbox("Heart Disease", ("No", "Yes"))
    ever_married = st.sidebar.selectbox("Ever Married", ("No", "Yes"))
    work_type = st.sidebar.selectbox("Work Type", ("Private", "Self-employed", "Govt_job", "Children", "Never_worked"))
    residence_type = st.sidebar.selectbox("Residence Type", ("Urban", "Rural"))
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50, 300, 100)
    bmi = st.sidebar.slider("BMI", 10, 60, 25)
    smoking_status = st.sidebar.selectbox("Smoking Status", ("never smoked", "formerly smoked", "smokes", "Unknown"))

    data = {
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    return pd.DataFrame([data])

input_df = user_input()

# One-hot encode input and match feature columns
input_encoded = pd.get_dummies(input_df)
df_encoded = pd.get_dummies(df.drop("stroke", axis=1))

# Ensure all required columns are present
for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[feature_columns]

# Predict
if st.button("Predict Stroke Risk"):
    prediction_proba = model.predict_proba(input_encoded)[0][1]
    prediction = int(prediction_proba >= best_threshold)

    st.subheader("Prediction:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke! (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Low Risk of Stroke (Probability: {prediction_proba:.2f})")

    # -------------- Dynamic Visualization --------------
    st.markdown("### Your Health vs Population Distribution")

    # Prepare visual data
    input_viz = input_df.copy()
    input_viz["Source"] = "You"
    df_temp = df.copy()
    df_temp["Source"] = "Population"

    # Columns to visualize
    numeric_features = ["age", "avg_glucose_level", "bmi"]
    df_viz = pd.concat([
        df_temp[numeric_features + ["Source"]],
        input_viz[numeric_features + ["Source"]]
    ], ignore_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = plt.figure()
        sns.kdeplot(data=df_viz, x="age", hue="Source", fill=True)
        plt.title("Age Distribution")
        st.pyplot(fig1)

        fig2 = plt.figure()
        sns.kdeplot(data=df_viz, x="bmi", hue="Source", fill=True)
        plt.title("BMI Distribution")
        st.pyplot(fig2)

    with col2:
        fig3 = plt.figure()
        sns.kdeplot(data=df_viz, x="avg_glucose_level", hue="Source", fill=True)
        plt.title("Glucose Level Distribution")
        st.pyplot(fig3)

# ... (Keep the imports and all previous code exactly as you pasted)


    # -------------- Patient Health Overview (Improved) --------------
    st.markdown("### 🩺 Patient Health Overview")

    def get_health_status(value, stroke_series, label):
        q25 = stroke_series.quantile(0.25)
        q75 = stroke_series.quantile(0.75)
        if value < q25:
            return f"🔵 Your **{label}** is lower than typical stroke patients."
        elif value > q75:
            return f"🔴 Your **{label}** is higher than typical stroke patients."
        else:
            return f"🟡 Your **{label}** is within the mid-range of stroke patients."

    # Individual values
    age_val = input_df['age'].values[0]
    bmi_val = input_df['bmi'].values[0]
    glucose_val = input_df['avg_glucose_level'].values[0]

    # Stroke patient population
    stroke_df = df[df["stroke"] == 1]
    age_stroke = stroke_df['age']
    bmi_stroke = stroke_df['bmi']
    glucose_stroke = stroke_df['avg_glucose_level']

    # Display based on stroke cases
    st.write(get_health_status(age_val, age_stroke, "age"))
    st.write(get_health_status(bmi_val, bmi_stroke, "BMI"))
    st.write(get_health_status(glucose_val, glucose_stroke, "glucose level"))

    # Contextual message
    if prediction == 1:
        st.warning("⚠️ You are predicted to be at **high risk of stroke**. The above analysis compares you with actual stroke patients.")
    else:
        st.success("✅ You are predicted to be at **low risk of stroke**. Your values are still compared with stroke-prone ranges.")

st.markdown("---")
st.markdown("Developed by **Rahul and Rakesh**")
