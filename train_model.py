# ============================================================
# M.TECH (AIML) - MACHINE LEARNING ASSIGNMENT 2
# STUDENT NAME: PRASAD V.S.
# BITS ID: 2026AJ05504
# File: train_model.py
#
# Objective:
# 1. Load the UCI Breast Cancer Wisconsin dataset.
# 2. Split the dataset into training and testing data.
# 3. Implement five classification algorithms.
# 4. Calculate six evaluation metrics for each model.
# 5. Save trained models using Joblib.
# 6. Export test_data.csv for the Streamlit application.
# ============================================================

import os
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import the five classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report
)

# ============================================================
# STEP 1: CREATE PROJECT DIRECTORIES
# ============================================================

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# STEP 2: LOAD THE DATASET
# ============================================================

# Load the Wisconsin Breast Cancer dataset from Scikit-learn.
# This is the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

dataset = load_breast_cancer()

# Convert the feature matrix into a Pandas DataFrame.
X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

# Convert the target into a Pandas Series.
# 0 = Malignant
# 1 = Benign
y = pd.Series(dataset.target, name="target")

# Combine input features and target for inspection.
df = X.copy()
df["target"] = y

print("\n========== DATASET INFORMATION ==========")

print("Dataset shape:", df.shape)
print("Number of input features:", X.shape[1])
print("Number of instances:", X.shape[0])

print("\nTarget distribution:")
print(y.value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())

# ============================================================
# STEP 3: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

# 80% of the observations are used for training.
# 20% of the observations are reserved for testing.
#
# stratify=y maintains approximately the same class
# proportions in both training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# Export the independent test data.
# The target column is included because the app needs
# actual labels to calculate evaluation metrics.

test_data = X_test.copy()
test_data["target"] = y_test

test_data.to_csv("test_data.csv", index=False)

# ============================================================
# STEP 4: DEFINE THE FIVE CLASSIFICATION MODELS
# ============================================================

# Each model is stored in a dictionary.
# Pipelines are used wherever feature scaling is needed.

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42
        ))
    ])
}

# ============================================================
# STEP 5: TRAIN MODELS AND CALCULATE EVALUATION METRICS
# ============================================================

results = []

# Dictionary containing filenames for the saved models.
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

for model_name, model in models.items():

    print("\n========================================")
    print("Training:", model_name)
    print("========================================")

    # Train using only training data.
    model.fit(X_train, y_train)

    # Predict class labels for test observations.
    y_pred = model.predict(X_test)

    # Calculate probability of the positive class (Benign = 1).
    y_probability = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # CALCULATE SIX EVALUATION METRICS
   # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # --------------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------------

    filename = model_files[model_name]

    joblib.dump(
        model,
        os.path.join("model", filename)
    )

    print("Accuracy:", round(accuracy, 4))
    print("AUC:", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("MCC:", round(mcc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Malignant", "Benign"],
            zero_division=0
        )
    )

    print("Saved model:", filename)

# ============================================================
# STEP 6: CREATE MODEL COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

# Round metric values to four decimal places.
results_df.iloc[:, 1:] = results_df.iloc[:, 1:].round(4)

# Export comparison table.
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n========== FINAL MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

# ============================================================
# STEP 7: IDENTIFY THE BEST MODEL
# ============================================================

# MCC is used as the primary comparison criterion here.
# It considers all four confusion-matrix outcomes.

best_model = results_df.loc[
    results_df["MCC"].idxmax()
]

print("\n========== BEST MODEL ==========")

print("Best model:", best_model["ML Model Name"])
print("Best MCC:", best_model["MCC"])

print("\nTraining and evaluation completed successfully!")

The information contained in this electronic communication is intended solely for the individual(s) or entity to which it is addressed. It may contain proprietary, confidential and/or legally privileged information. Any review, retransmission, dissemination, printing, copying or other use of, or taking any action in reliance on the contents of this information by person(s) or entities other than the intended recipient is strictly prohibited and may be unlawful. If you have received this communication in error, please notify us by responding to this email or telephone and immediately and permanently delete all copies of this message and any attachments from your system(s). The contents of this message do not necessarily represent the views or policies of BITS Pilani.
