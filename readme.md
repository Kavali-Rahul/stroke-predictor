# 🧠 Stroke Risk Prediction Web App

This project is a machine learning-based web app to predict the risk of stroke in a patient based on health and lifestyle factors. It also provides visual comparisons of the patient's health data with the general population.

## Features

- Predict stroke risk using a Random Forest model
- Compares:
  - Age
  - BMI
  - Glucose levels
- Interactive charts for visualization
- Shows patient health status analysis
- Easy to use web interface (built with Streamlit)

## How to Use

1. Clone this repo or upload it to Streamlit Cloud
2. Run the app:
   ```bash
   streamlit run stroke_app.py

stroke_prediction_project/
│
├── stroke_app.py                # Streamlit app
├── stroke_data.csv              # Dataset used for visualization
├── saved_model/
│   ├── stroke_rf_model.pkl      # Trained model
│   ├── feature_columns.pkl      # Feature list
│   └── best_threshold.txt       # Optimal threshold value

## dataset
This app uses a modified version of the Stroke Prediction Dataset from Kaggle.

📊 Output Example
🔴 High Risk of Stroke (Probability: 0.78)

🟢 Your BMI is in the average range

🔴 Your glucose level is higher than most people



**DEVELOPED BY KAVALI RAHUL**