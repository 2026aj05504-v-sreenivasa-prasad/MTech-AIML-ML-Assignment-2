# ============================================================
# M.TECH (AIML) - MACHINE LEARNING ASSIGNMENT 2
# STUDENT NAME: PRASAD V.S.
# BITS ID: 2026AJ05504
# File: app.py
#
# Interactive Machine Learning Model Evaluation Dashboard
#
# Features:
# 1. Upload test dataset in CSV format.
# 2. Select one of five trained ML models.
# 3. Compare all five models.
# 4. Display six evaluation metrics.
# 5. Display confusion matrix.
# 6. Display classification report.
# 7. Download model comparison results.
#
# IMPORTANT:
# The app loads already-trained models.
# It DOES NOT retrain models on uploaded test data.
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


# ============================================================
# STEP 1: STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STEP 2: CUSTOM CSS STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #243B53;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 15px;
        color: #627D98;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 600;
        color: #243B53;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .info-text {
        font-size: 14px;
        color: #486581;
    }

    div[data-testid="stMetric"] {
        background-color: #F5F8FA;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #E1E8ED;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STEP 3: PROJECT DIRECTORIES AND MODEL FILES
# ============================================================

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

MODEL_DIRECTORY = os.path.join(BASE_DIRECTORY, "model")
RESULTS_DIRECTORY = os.path.join(BASE_DIRECTORY, "results")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

MODEL_NAMES = list(MODEL_FILES.keys())

CLASS_NAMES = ["Malignant", "Benign"]
TARGET_COLUMN = "target"


# ============================================================
# STEP 4: LOAD PREVIOUSLY TRAINED MODELS
# ============================================================

# Display dashboard heading BEFORE loading models
st.markdown(
    '<div class="main-title">'
    '🎗️ Breast Cancer Classification Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'M.Tech (AIML) | Machine Learning Assignment 2'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Initializing the dashboard and loading five previously "
    "trained machine learning models. Please wait..."
)


@st.cache_resource
def load_models():

    loaded_models = {}

    print("STARTING MODEL LOADING...", flush=True)

    for model_name, filename in MODEL_FILES.items():

        model_path = os.path.join(
            MODEL_DIRECTORY,
            filename
        )

        print(
            f"Loading {model_name}: {model_path}",
            flush=True
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        loaded_models[model_name] = joblib.load(
            model_path
        )

        print(
            f"SUCCESS: {model_name} loaded.",
            flush=True
        )

    print("ALL FIVE MODELS LOADED SUCCESSFULLY!", flush=True)

    return loaded_models


try:

    with st.spinner(
        "Loading all five trained ML models..."
    ):

        models = load_models()

    st.success(
        "All five trained ML models loaded successfully!"
    )

except Exception as error:

    print(
        f"MODEL LOADING ERROR: {repr(error)}",
        flush=True
    )

    st.error(
        f"Unable to load trained models: {error}"
    )

    st.info(
        "Please verify that all five Joblib files exist "
        "in the model folder and that their library versions "
        "are compatible."
    )

    st.stop()

# ============================================================
# STEP 5: APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎗️ Breast Cancer Classification Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'M.Tech (AIML) | Machine Learning Assignment 2'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    """
    An interactive machine learning application for evaluating
    five classification algorithms using the Breast Cancer
    Wisconsin Diagnostic dataset.

    Upload an independent test dataset, select a classifier,
    and visualize its classification performance.
    """
)

st.info(
    "All five classifiers are pre-trained and loaded using "
    "Joblib. Uploaded test data is used only for evaluation. "
    "No model retraining takes place inside this application."
)


# ============================================================
# STEP 6: SIDEBAR CONFIGURATION
# ============================================================

st.sidebar.title("⚙️ Model Configuration")

st.sidebar.write(
    "Configure your classification experiment."
)

st.sidebar.divider()


# 6.1 SELECT CLASSIFIER

st.sidebar.subheader("Choose Classifier")

selected_model_name = st.sidebar.selectbox(
    "Classification Algorithm",
    MODEL_NAMES,
    index=0
)

selected_model = models[selected_model_name]


# 6.2 DISPLAY SAVED MODEL PARAMETERS

st.sidebar.subheader("Saved Model Parameters")

st.sidebar.caption(
    "The following parameters are informational. "
    "The saved model is not modified or retrained."
)

classifier = getattr(
    selected_model,
    "named_steps",
    {}
)

if hasattr(classifier, "get"):
    classifier = classifier.get("classifier", selected_model)
else:
    classifier = selected_model

model_parameters = classifier.get_params()

if selected_model_name == "Logistic Regression":

    st.sidebar.caption(
        f"Regularization parameter C: "
        f"{model_parameters.get('C', 'N/A')}"
    )

    st.sidebar.caption(
        f"Maximum iterations: "
        f"{model_parameters.get('max_iter', 'N/A')}"
    )

elif selected_model_name == "Decision Tree":

    st.sidebar.caption(
        f"Maximum depth: "
        f"{model_parameters.get('max_depth', 'N/A')}"
    )

elif selected_model_name == "KNN":

    st.sidebar.caption(
        f"Number of neighbours: "
        f"{model_parameters.get('n_neighbors', 'N/A')}"
    )

elif selected_model_name == "Naive Bayes":

    st.sidebar.caption("Algorithm: Gaussian Naive Bayes")

elif selected_model_name == "Support Vector Machine":

    st.sidebar.caption(
        f"Kernel: {model_parameters.get('kernel', 'N/A')}"
    )

    st.sidebar.caption(
        f"Regularization parameter C: "
        f"{model_parameters.get('C', 'N/A')}"
    )

st.sidebar.caption(
    "These are the saved training parameters. "
    "Changing them requires a separate training run."
)


# 6.3 UPLOAD TEST DATASET

st.sidebar.divider()
st.sidebar.subheader("Upload Test Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# 6.4 CLASSIFICATION THRESHOLD

st.sidebar.divider()
st.sidebar.subheader("Classification Threshold")

threshold = st.sidebar.slider(
    "Probability threshold for Benign class",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.caption(
    "Predicted probability >= threshold: Benign (1). "
    "Otherwise: Malignant (0)."
)


# 6.5 SELECT EVALUATION PLOTS

st.sidebar.divider()
st.sidebar.subheader("What Metrics to Plot?")

selected_plots = st.sidebar.multiselect(
    "Select visualizations",
    [
        "Confusion Matrix",
        "ROC Curve",
        "Precision-Recall Curve"
    ],
    default=[
        "Confusion Matrix",
        "ROC Curve",
        "Precision-Recall Curve"
    ]
)


# SIDEBAR INFORMATION

st.sidebar.divider()
st.sidebar.subheader("Dataset Information")

st.sidebar.write("Dataset: Breast Cancer Wisconsin")
st.sidebar.write("Instances: 569")
st.sidebar.write("Input features: 30")
st.sidebar.write("Classes: 2")
st.sidebar.write("0: Malignant")
st.sidebar.write("1: Benign")

st.sidebar.divider()

st.sidebar.caption(
    "M.Tech AIML - Machine Learning Assignment 2"
)

# ============================================================
# STEP 7: DISPLAY AVAILABLE CLASSIFIERS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '1. Available Classification Models'
    '</div>',
    unsafe_allow_html=True
)

model_columns = st.columns(5)

for column, model_name in zip(model_columns, MODEL_NAMES):

    with column:
        st.markdown(
            f"""
            <div style="
                background-color: #F5F8FA;
                padding: 15px 8px;
                border-radius: 10px;
                border: 1px solid #E1E8ED;
                text-align: center;
                min-height: 100px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: 600;
                color: #243B53;
                overflow-wrap: anywhere;
            ">
                {model_name}
            </div>
            """,
            unsafe_allow_html=True
        )
# ============================================================
# STEP 8: DATASET UPLOAD AND VALIDATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '2. Test Dataset'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload the independent test_data.csv file "
    "generated by train_model.py."
)

if uploaded_file is None:

    st.warning(
        "Please upload test_data.csv using the sidebar "
        "to begin model evaluation."
    )

    st.stop()


try:
    test_df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"Unable to read uploaded CSV: {error}"
    )

    st.stop()


# 8.1 CHECK EMPTY DATASET

if test_df.empty:

    st.error("The uploaded dataset is empty.")
    st.stop()


# 8.2 GET EXPECTED FEATURE NAMES

try:

    expected_features = list(
        models["Logistic Regression"].feature_names_in_
    )

except AttributeError:

    st.error(
        "The Logistic Regression model does not contain "
        "feature_names_in_. Please verify your trained model."
    )

    st.stop()


required_columns = expected_features + [TARGET_COLUMN]


# 8.3 CHECK MISSING COLUMNS

missing_columns = [
    column
    for column in required_columns
    if column not in test_df.columns
]

if missing_columns:

    st.error(
        "The uploaded dataset is missing required columns:"
    )

    st.write(missing_columns)

    st.stop()


# 8.4 DISPLAY DATASET PREVIEW

st.success("Test dataset uploaded successfully.")

st.subheader("Uploaded Dataset Preview")

st.dataframe(
    test_df.head(10),
    use_container_width=True
)

dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

dataset_col1.metric(
    "Total Records",
    len(test_df)
)

dataset_col2.metric(
    "Input Features",
    len(expected_features)
)

dataset_col3.metric(
    "Target Classes",
    test_df[TARGET_COLUMN].nunique()
)


# ============================================================
# STEP 9: PREPARE AND VALIDATE INPUT DATA
# ============================================================

X_test = test_df[expected_features].copy()

y_test = test_df[TARGET_COLUMN].copy()


# 9.1 NUMERIC VALIDATION

try:

    X_test = X_test.apply(
        pd.to_numeric,
        errors="raise"
    )

    y_test = pd.to_numeric(
        y_test,
        errors="raise"
    )

except (ValueError, TypeError):

    st.error(
        "All input features and target values must be numeric."
    )

    st.stop()


# 9.2 CHECK MISSING VALUES

if X_test.isnull().any().any() or y_test.isnull().any():

    st.error(
        "The uploaded dataset contains missing values."
    )

    st.stop()


# 9.3 CHECK INFINITE VALUES

if not np.isfinite(
    X_test.to_numpy(dtype=float)
).all():

    st.error(
        "Input features contain infinite or invalid values."
    )

    st.stop()


# 9.4 CHECK TARGET ENCODING

if not set(y_test.unique()).issubset({0, 1}):

    st.error(
        "Invalid target values. Expected 0 = Malignant "
        "and 1 = Benign."
    )

    st.stop()


if set(y_test.unique()) != {0, 1}:

    st.error(
        "Both classes must be present in the uploaded "
        "test dataset to calculate ROC-AUC."
    )

    st.stop()


# Ensure integer target labels

y_test = y_test.astype(int)

st.success(
    "Dataset validation completed successfully."
)


# ============================================================
# STEP 10: MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    classification_threshold
):

    # Check probability prediction support

    if not hasattr(model, "predict_proba"):

        raise AttributeError(
            "The selected model does not support predict_proba(). "
            "For SVM, train the model with probability=True."
        )

    # Obtain class-1 (Benign) probabilities

    if hasattr(model, "classes_"):

        classes = list(model.classes_)

    elif hasattr(model, "named_steps"):

        classifier = model.named_steps["classifier"]
        classes = list(classifier.classes_)

    else:

        raise ValueError(
            "Unable to identify the model's class labels."
        )

    if 1 not in classes:

        raise ValueError(
            "The model does not contain class 1 (Benign)."
        )

    class_one_index = classes.index(1)

    probabilities = model.predict_proba(X)[
        :, class_one_index
    ]

    # Apply adjustable classification threshold

    predictions = (
        probabilities >= classification_threshold
    ).astype(int)

    # Calculate evaluation metrics

    accuracy = accuracy_score(
        y,
        predictions
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y,
        predictions
    )

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "Predictions": predictions,
        "Probabilities": probabilities
    }


# ============================================================
# STEP 11: EVALUATE ALL FIVE CLASSIFICATION MODELS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '3. Model Performance Comparison'
    '</div>',
    unsafe_allow_html=True
)

evaluation_results = []

all_predictions = {}

all_probabilities = {}

for model_name, model in models.items():

    evaluation = evaluate_model(
        model,
        X_test,
        y_test,
        threshold
    )

    all_predictions[model_name] = evaluation[
        "Predictions"
    ]

    all_probabilities[model_name] = evaluation[
        "Probabilities"
    ]

    evaluation_results.append({

        "ML Model Name": model_name,

        "Accuracy": evaluation["Accuracy"],

        "AUC": evaluation["AUC"],

        "Precision": evaluation["Precision"],

        "Recall": evaluation["Recall"],

        "F1": evaluation["F1"],

        "MCC": evaluation["MCC"]

    })


results_df = pd.DataFrame(evaluation_results)

# ============================================================
# STEP 12: DISPLAY COMPARISON TABLE
# ============================================================

st.subheader(
    "Performance of All Five Classification Models"
)

st.caption(
    "Precision, Recall and F1 use Benign (class 1) "
    "as the positive class. AUC uses predicted probabilities."
)

st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "MCC": "{:.4f}"
    }),
    use_container_width=True
)


# ============================================================
# STEP 13: IDENTIFY BEST PERFORMING MODEL
# ============================================================

best_index = results_df["MCC"].idxmax()

best_result = results_df.loc[best_index]

st.subheader(
    "Best Performing Classification Model"
)

best_col1, best_col2, best_col3 = st.columns(3)

best_col1.metric(
    "Best Model",
    best_result["ML Model Name"]
)

best_col2.metric(
    "Best Accuracy",
    f"{best_result['Accuracy']:.4f}"
)

best_col3.metric(
    "Best MCC",
    f"{best_result['MCC']:.4f}"
)

st.caption(
    "The best model is selected using the highest "
    "Matthews Correlation Coefficient (MCC)."
)


# ============================================================
# STEP 14: MODEL COMPARISON CHART
# ============================================================

st.divider()

st.subheader(
    "Visual Comparison of All Five Models"
)

chart_data = results_df.set_index(
    "ML Model Name"
)[
    [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
]

st.bar_chart(
    chart_data,
    use_container_width=True
)


# ============================================================
# STEP 15: INDIVIDUAL MODEL EVALUATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '4. Individual Model Evaluation'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    f"Selected classifier: {selected_model_name}"
)

selected_predictions = all_predictions[
    selected_model_name
]

selected_probabilities = all_probabilities[
    selected_model_name
]

selected_result = results_df[
    results_df["ML Model Name"] == selected_model_name
].iloc[0]


# ============================================================
# STEP 16: DISPLAY SIX EVALUATION METRICS
# ============================================================

st.subheader(
    f"Evaluation Metrics - {selected_model_name}"
)

metric_columns_1 = st.columns(3)

metric_columns_1[0].metric(
    "Accuracy",
    f"{selected_result['Accuracy']:.4f}"
)

metric_columns_1[1].metric(
    "AUC Score",
    f"{selected_result['AUC']:.4f}"
)

metric_columns_1[2].metric(
    "Precision",
    f"{selected_result['Precision']:.4f}"
)


metric_columns_2 = st.columns(3)

metric_columns_2[0].metric(
    "Recall",
    f"{selected_result['Recall']:.4f}"
)

metric_columns_2[1].metric(
    "F1 Score",
    f"{selected_result['F1']:.4f}"
)

metric_columns_2[2].metric(
    "MCC Score",
    f"{selected_result['MCC']:.4f}"
)


# ============================================================
# STEP 17: DISPLAY CLASS-SPECIFIC PERFORMANCE
# ============================================================

st.subheader(
    "Class-wise Performance"
)

class_report = classification_report(
    y_test,
    selected_predictions,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

class_report_df = pd.DataFrame(
    class_report
).transpose()

st.dataframe(
    class_report_df.style.format("{:.4f}"),
    use_container_width=True
)

st.info(
    f"Malignant-class recall: "
    f"{class_report['Malignant']['recall']:.4f}. "
    "This metric indicates the proportion of actual malignant "
    "observations correctly classified as malignant."
)


# ============================================================
# STEP 18: CONFUSION MATRIX
# ============================================================

if "Confusion Matrix" in selected_plots:

    st.divider()

    st.subheader(
        f"Confusion Matrix - {selected_model_name}"
    )

    cm = confusion_matrix(
        y_test,
        selected_predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=1,
        linecolor="black",
        cbar=True,
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=11
    )

    ax.set_ylabel(
        "True Label",
        fontsize=11
    )

    ax.set_title(
        f"Confusion Matrix - {selected_model_name}",
        fontsize=13
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

# ============================================================
# STEP 19: ROC CURVE
# ============================================================

if "ROC Curve" in selected_plots:

    st.divider()

    st.subheader(
        f"ROC Curve - {selected_model_name}"
    )

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        selected_probabilities,
        pos_label=1
    )

    auc_value = roc_auc_score(
        y_test,
        selected_probabilities
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        linewidth=2,
        label=f"ROC Curve (AUC = {auc_value:.4f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Receiver Operating Characteristic Curve"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.3
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# STEP 20: PRECISION-RECALL CURVE
# ============================================================

if "Precision-Recall Curve" in selected_plots:

    st.divider()

    st.subheader(
        f"Precision-Recall Curve - {selected_model_name}"
    )

    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_test,
            selected_probabilities,
            pos_label=1
        )
    )

    average_precision = average_precision_score(
        y_test,
        selected_probabilities
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        recall_values,
        precision_values,
        color="green",
        linewidth=2,
        label=f"Average Precision = {average_precision:.4f}"
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Precision-Recall Curve"
    )

    ax.legend(
        loc="best"
    )

    ax.grid(
        alpha=0.3
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# STEP 21: DOWNLOAD EVALUATION RESULTS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '5. Download Evaluation Results'
    '</div>',
    unsafe_allow_html=True
)

csv_results = results_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Model Comparison CSV",
    data=csv_results,
    file_name="model_comparison.csv",
    mime="text/csv"
)


# ============================================================
# STEP 22: SAVE RESULTS TO PROJECT DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True
)

results_path = os.path.join(
    RESULTS_DIRECTORY,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# STEP 23: APPLICATION STATUS
# ============================================================

st.success(
    "All five trained classification models "
    "have been evaluated successfully!"
)


# ============================================================
# STEP 24: APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "M.Tech (AIML) | Machine Learning Assignment 2"
)

st.caption(
    "Dataset: Breast Cancer Wisconsin Diagnostic Dataset"
)

st.caption(
    "Algorithms: Logistic Regression, Decision Tree, "
    "KNN, Gaussian Naive Bayes and Support Vector Machine"
)

st.caption(
    "DISCLAIMER: This is educational demonstration only (As part of BITS MTech AI/ML Project). "
    "This application is not intended for medical diagnosis."
)

The information contained in this electronic communication is intended solely for the individual(s) or entity to which it is addressed. It may contain proprietary, confidential and/or legally privileged information. Any review, retransmission, dissemination, printing, copying or other use of, or taking any action in reliance on the contents of this information by person(s) or entities other than the intended recipient is strictly prohibited and may be unlawful. If you have received this communication in error, please notify us by responding to this email or telephone and immediately and permanently delete all copies of this message and any attachments from your system(s). The contents of this message do not necessarily represent the views or policies of BITS Pilani.
