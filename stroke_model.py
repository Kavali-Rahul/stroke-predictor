import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.combine import SMOTEENN
import joblib
import os

# ---------------------- Load dataset ----------------------
df = pd.read_csv("stroke_data.csv")
df.drop(columns=['id'], inplace=True, errors='ignore')
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())  # avoid chained assignment warning

# ---------------------- Feature Engineering ----------------------
categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('stroke', axis=1)
y = df['stroke']
feature_columns = list(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------- SMOTEENN for balancing ----------------------
print(f"Before SMOTEENN: {y_train.value_counts().to_dict()}")
smenn = SMOTEENN(random_state=42)
X_res, y_res = smenn.fit_resample(X_train, y_train)
print(f"After SMOTEENN: {pd.Series(y_res).value_counts().to_dict()}")

# ---------------------- Random Forest Hyperparameter Tuning ----------------------
param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt','log2']
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid, scoring='f1', cv=3, n_jobs=-1
)
grid.fit(X_res, y_res)
best_rf_model = grid.best_estimator_
print(f"✅ Best Random Forest Parameters: {grid.best_params_}")

# ---------------------- Feature Importance (Top 20) ----------------------
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [feature_columns[i] for i in indices[:20]]

# ---------------------- Align test set with training features ----------------------
for col in X_res.columns:
    if col not in X_test.columns:
        X_test[col] = 0

X_test_aligned = X_test[X_res.columns]

X_train_top = X_res[top_features]
X_test_top = X_test_aligned[top_features]

# ---------------------- Train Random Forest on top features ----------------------
best_rf_model.fit(X_train_top, y_res)

# ---------------------- Fixed Threshold Prediction ----------------------
fixed_threshold = 0.41
rf_probs = best_rf_model.predict_proba(X_test_top)[:,1]
y_pred = (rf_probs >= fixed_threshold).astype(int)

# ---------------------- Evaluation ----------------------
print(f"\n✅ Fixed Threshold: {fixed_threshold}")
print("🎯 Accuracy:", round(accuracy_score(y_test, y_pred),4))
print("📄 Classification Report:\n", classification_report(y_test, y_pred))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------- Save Model and Features ----------------------
os.makedirs("saved_model", exist_ok=True)
joblib.dump(best_rf_model, "saved_model/stroke_rf_model.pkl")
with open("saved_model/best_threshold.txt","w") as f:
    f.write(str(fixed_threshold))
joblib.dump(top_features,"saved_model/feature_columns.pkl")

print("💾 Model, threshold, and top features saved in 'saved_model/' folder.")
