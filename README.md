# M.Tech (AIML) - Machine Learning Assignment 2
**Five Classification Models and Streamlit Deployment**
## STUDENT NAME: PRASAD V.S.
## BITS ID: 2026AJ05504

## Project Title
Comparative Analysis of Five Machine Learning Classification
Algorithms Using the Breast Cancer Wisconsin Dataset

## A. Problem Statement

The objective of this project is to implement and evaluate
five Machine Learning classification algorithms using a
publicly available classification dataset.

The five algorithms implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine

The models are evaluated using six performance metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

An interactive Streamlit application is also developed
to allow users to upload test data, select classifiers,
and compare model performance.

The trained models are saved and loaded by the Streamlit
application without retraining on uploaded test data.

## B. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)

**Dataset Source:** UCI Machine Learning Repository

**Official Dataset Link:**
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Dataset characteristics:

| Attribute             | Description                  |
|-----------------------|------------------------------|
| Total Instances       | 569                          |
| Input Features        | 30                           |
| Target Classes        | 2                            |
| Classification Type   | Binary                       |
| Data Type             | Numerical                    |
| Missing Values        | None in the standard dataset |

### Target Variable

The target variable represents the diagnosis:

- 0 = Malignant
- 1 = Benign

### Input Features

The dataset contains 30 numerical features describing
characteristics of cell nuclei.

Examples include:

- Radius mean
- Texture mean
- Perimeter mean
- Area mean
- Smoothness mean
- Compactness mean
- Concavity mean
- Symmetry mean

### Data Preprocessing

1. Loaded the dataset.
2. Separated input features and target variable.
3. Divided the data into training and testing datasets.
4. Used an 80:20 training-testing split.
5. Used stratified sampling to preserve class proportions.
6. Applied StandardScaler through pipelines for
   algorithms requiring feature scaling.
7. Saved the independent test dataset as test_data.csv.

The random state was fixed at 42 to ensure reproducibility.

