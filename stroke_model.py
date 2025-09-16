import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay

# Load dataset
df = pd.read_csv("stroke_data.csv")

# Split features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Apply SMOTE on preprocessed training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "LDA": LDA(),
    "QDA": QDA(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "MLP": MLPClassifier(max_iter=500),
    "Custom Ensemble": StackingClassifier(
        estimators=[
            ('catboost', CatBoostClassifier(verbose=0)),
            ('lgbm', LGBMClassifier()),
            ('logreg', LogisticRegression(max_iter=1000))
        ],
        final_estimator=LogisticRegression()
    )
}

# Train and evaluate
results = []
plt.figure(figsize=(12,8))
for i, (name, model) in enumerate(models.items()):
    print(f"Training {name}...")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_preprocessed)
    y_prob = model.predict_proba(X_test_preprocessed)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test_preprocessed)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    avg_acc_recall = (acc + rec) / 2
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Recall": rec,
        "Precision": prec,
        "F1 Score": f1,
        "AUC-ROC": auc,
        "Confusion Matrix": cm,
        "Avg_Acc_Recall": avg_acc_recall
    })
    
    # Plot ROC curve
    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, alpha=0.6, ax=plt.gca())

plt.title("ROC Curves for All Models")
plt.legend(loc="lower right")
plt.show()

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by F1 score for comparison table (keep full metrics)
results_df_sorted = results_df.sort_values(by="F1 Score", ascending=False)
print("\n=== Model Comparison ===")
print(results_df_sorted)

# Calculate average of Accuracy & Recall
results_df["Avg_Acc_Recall"] = (results_df["Accuracy"] + results_df["Recall"]) / 2

# Select best model based on Avg_Acc_Recall
best_model = results_df.loc[results_df["Avg_Acc_Recall"].idxmax()]
print("\nBest Model (by Avg Accuracy & Recall):")
print(best_model)