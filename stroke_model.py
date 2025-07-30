import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("stroke_data.csv")

# Drop 'id' column if present
df.drop(columns=['id'], inplace=True, errors='ignore')

# Fill missing BMI values
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df)

# Split into features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Save feature columns for later use in prediction
feature_columns = list(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ----------------------🔹 XGBoost Model -----------------------
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

print("\n🔹 Model: XGBoost")
print("🎯 Accuracy:", round(accuracy_score(y_test, xgb_preds), 4))
print("📄 Classification Report:\n", classification_report(y_test, xgb_preds))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))

# ----------------------🔹 CatBoost Model -----------------------
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)

print("\n🔹 Model: CatBoost")
print("🎯 Accuracy:", round(accuracy_score(y_test, cat_preds), 4))
print("📄 Classification Report:\n", classification_report(y_test, cat_preds))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, cat_preds))

# ----------------------🔹 RandomForest + Threshold Tuning -----------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Threshold tuning
thresholds = np.arange(0.10, 0.30, 0.03)
best_f1 = 0
best_threshold = 0.5
best_rf_model = None

print("\n🔹 RandomForest - Threshold Tuning")
for threshold in thresholds:
    preds = (rf_probs >= threshold).astype(int)
    f1 = f1_score(y_test, preds)
    print(f"Threshold: {threshold:.2f} | F1 Score: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_rf_model = rf_model

print(f"\n✅ Best Random Forest model saved with threshold {best_threshold:.2f}")

# ----------------------✅ Save Model, Threshold, Features -----------------------
os.makedirs("saved_model", exist_ok=True)

# Save best model
joblib.dump(best_rf_model, "saved_model/stroke_rf_model.pkl")

# Save best threshold
with open("saved_model/best_threshold.txt", "w") as f:
    f.write(str(best_threshold))

# Save feature columns
joblib.dump(feature_columns, "saved_model/feature_columns.pkl")

print("💾 Model, threshold, and feature columns saved in 'saved_model/' folder.")

