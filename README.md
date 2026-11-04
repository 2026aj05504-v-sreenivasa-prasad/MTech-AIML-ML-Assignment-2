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

## C. GitHub Repository Link

**GitHub Repository:**

https://github.com/2026aj05504-v-sreenivasa-prasad/MTech-AIML-ML-Assignment-2/

The repository contains:

- app.py
- train_model.py
- requirements.txt
- test_data.csv
- model/
- results/
- README.md

All five trained classification models are saved
using Joblib.

## D. Models Used and Evaluation Metrics

All five classifiers were trained using the same dataset
and evaluated using the same independent test dataset.

### 1. Logistic Regression

Logistic Regression is a supervised classification
algorithm that estimates the probability of a sample
belonging to a particular class.

### 2. Decision Tree Classifier

Decision Tree is a supervised learning algorithm that
classifies samples using a sequence of decision rules.

### 3. K-Nearest Neighbors Classifier

KNN classifies a new sample based on the classes of
its nearest neighboring observations.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier
based on Bayes' theorem and assumes Gaussian
distributions for input features within each class.

### 5. Support Vector Machine

SVM identifies an optimal decision boundary to
separate observations belonging to different classes.
The RBF kernel is used in this implementation.

### Model Comparison Table

<img width="797" height="314" alt="image" src="https://github.com/user-attachments/assets/f26761ed-6c67-498e-a480-95d72e95a102" />

# Observations About Model Performance

## 1. Logistic Regression

Logistic Regression demonstrated excellent performance across all six evaluation metrics. It achieved an accuracy of 0.9825 (98.25%), an AUC of 0.9954, precision of 0.9861, recall of 0.9861, an F1-score of 0.9861, and an MCC of 0.9623. These results indicate that the model is highly accurate and provides an excellent balance between precision and recall. Its very high AUC also indicates outstanding ability to distinguish between benign and malignant classes. Overall, Logistic Regression showed the best performance among the five classification models.

## 2. Decision Tree

The Decision Tree model achieved an accuracy of 0.9211 (92.11%), an AUC of 0.9163, precision of 0.9565, recall of 0.9167, an F1-score of 0.9362, and an MCC of 0.8341. Although it achieved reasonably high precision, its recall was comparatively lower, indicating that it missed more positive cases than the other high-performing models. It also recorded the lowest AUC and MCC among all five models. This suggests that the Decision Tree performed relatively less effectively in comparison with the other classification algorithms.

## 3. KNN (K-Nearest Neighbors)

The KNN model achieved an accuracy of 0.9561 (95.61%), an AUC of 0.9788, precision of 0.9589, recall of 0.9722, an F1-score of 0.9655, and an MCC of 0.9054. The model demonstrated good overall classification performance, with particularly high recall, indicating that it correctly identified a large proportion of positive cases. Its F1-score also reflects a good balance between precision and recall. However, its performance was slightly lower than Logistic Regression and Support Vector Machine across most evaluation metrics.

## 4. Naive Bayes

The Naive Bayes model achieved an accuracy of 0.9298 (92.98%), an AUC of 0.9868, precision of 0.9444, recall of 0.9444, an F1-score of 0.9444, and an MCC of 0.8492. The model demonstrated a relatively high AUC, indicating good class discrimination based on predicted probabilities. Its precision and recall were equal, showing a balanced classification performance. However, its accuracy, F1-score, and MCC were lower than those of KNN, Logistic Regression, and Support Vector Machine. Therefore, Naive Bayes provided reasonable performance but was not the best-performing model.

## 5. Support Vector Machine (SVM)

The Support Vector Machine model achieved an accuracy of 0.9825 (98.25%), an AUC of 0.9950, precision of 0.9861, recall of 0.9861, an F1-score of 0.9861, and an MCC of 0.9623. Its performance was almost identical to Logistic Regression across all six evaluation metrics. It achieved the same accuracy, precision, recall, F1-score, and MCC as Logistic Regression, with only a marginally lower AUC. These results indicate that SVM is also highly effective at classifying the data and distinguishing between the two classes.

## Overall Winner

Logistic Regression is the best-performing model among the five classification algorithms.

It achieved the highest accuracy of 98.25%, tied with Support Vector Machine, demonstrating excellent overall classification performance. It also achieved the highest AUC of 0.9954, indicating a slightly better ability to distinguish between the two classes compared with the other models.

Furthermore, Logistic Regression achieved a precision of 0.9861 and a recall of 0.9861, indicating that it correctly identifies positive cases while maintaining a very low proportion of false positive predictions. Its F1-score of 0.9861 confirms an excellent balance between precision and recall. Additionally, its MCC of 0.9623 indicates a very strong correlation between the actual and predicted classifications.

Although Support Vector Machine achieved identical values for accuracy, precision, recall, F1-score, and MCC, Logistic Regression performed slightly better in terms of AUC (0.9954 compared with 0.9950).

Therefore, considering all six evaluation metrics, Logistic Regression is selected as the overall best-performing classification model, with Support Vector Machine being a very close second. Both models demonstrated excellent classification capabilities, but Logistic Regression achieved the highest AUC and the best overall results in this comparison.

### NOTE: 
Logistic Regression and SVM are extremely close in performance. The difference in AUC is only 0.0004, so it is more accurate academically to conclude that Logistic Regression as the marginal winner and not claiming it is significantly better than SVM.

## E. Streamlit Application

The project includes an interactive Streamlit
application for evaluating the trained models.

### Application Features

1. Upload test dataset in CSV format.
2. Preview uploaded test data.
3. Select one of five classification models.
4. Compare evaluation results of all five models.
5. Display Accuracy, AUC, Precision, Recall, F1 and MCC.
6. Display confusion matrix.
7. Display classification report.
8. Download model comparison results.

### Live Streamlit Application

https://2026aj05504-v-sreenivasa-prasad-mtech-aiml-ml-assign-app-crcpgf.streamlit.app/

## F. Implementation Details

### Training and Testing

- Training split: 80%
- Testing split: 20%
- Random state: 42
- Stratified sampling: Yes

### Project structure
Below is the directory structure of the project

<img width="441" height="443" alt="image" src="https://github.com/user-attachments/assets/f6d46b0b-55e9-4d52-a5a1-0c7d8e2713cf" />
 
### Model Saving

All trained models are stored in the model directory
using Joblib.

### Test Data

The test_data.csv file contains all 30 input features
and the target variable.

The Streamlit application loads the previously saved
models and evaluates them on uploaded test data.

#### Note: No model retraining takes place inside the application.

## G. How to Run the Project

### Step 1: Clone the repository

git clone MTech-AIML-ML-Assignment-2

### Step 2: Enter the project directory

cd MTech-AIML-ML-Assignment-2

### Step 3: Install dependencies

pip install -r requirements.txt

### Step 4: Train the models

python train_model.py

### Step 5: Launch the Streamlit application

streamlit run app.py

### Step 6: Upload test_data.csv

Upload the generated test_data.csv file to evaluate
the five trained classification models.

## H. Conclusion

This project demonstrates an end-to-end Machine
Learning classification workflow.

Five supervised classification algorithms were
implemented using the Breast Cancer Wisconsin
Diagnostic dataset.

The models were evaluated using six performance
metrics and their performance was compared.

The trained models were saved using Joblib and
integrated into an interactive Streamlit application.

The application allows users to upload test datasets,
evaluate different classifiers, view confusion matrices,
and compare model performance.

The project demonstrates practical understanding of
Machine Learning implementation, model evaluation,
visualization and deployment.

## I. Project Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- GitHub
- Streamlit Community Cloud
