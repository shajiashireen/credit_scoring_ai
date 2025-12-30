# Credit Scoring System 

# Project Overview
This project implements an Explainable AI (XAI) based Credit Scoring System using machine learning.
It predicts whether a loan should be approved or rejected and explains the decision using SHAP and LIME.

The application is built using Streamlit and allows users to upload data, view predictions,
and understand model reasoning through visual explanations.


# Objectives
- Train a machine learning model on tabular credit data
- Predict loan approval outcomes
- Explain predictions using SHAP and LIME
- Provide both global and local interpretability
- Build an interactive dashboard


# Technologies Used
- Python
- scikit-learn
- SHAP
- LIME
- Streamlit
- Pandas, NumPy
- Matplotlib
- Joblib

# Project Structure

credit_scoring_xai/
│
├── app.py
├── model/
│   └── model.pkl
├── demo_data/
│   └── loan_demo.csv
├── screenshots/
│   ├── prediction.png
│   ├── shap_global.png
│   ├── shap_local.png
│   └── lime_explanation.png
└── README.md


# Dataset Description
The data applied to this project will be loan application records that are taken with the view to make a prediction on whether a loan is approved or not. One line is associated with one applicant and one column is associated with one of the demographic, financial or credit-related factors that affect the loan decision.

It is generally used in credit risk and loan approval prediction and is appropriate to illustrate explainable machine learning methods.
The dataset contains the following features:

- Loan_ID
- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (Target)

# Data Preprocessing Steps

- The preprocessing of the model in the steps that precede training are as follows:

- Remove identifier column

- LoanID is eliminated as it is not used to predict.

# Handle missing values

- The median is used to fill in the numerical features.

- Categorical characteristics are filled with the most common value (mode).

- This enables machine learning models to work with categorical values as numbers.

# Feature scaling

- The StandardScaler is utilized to standardize the numerical features.

- This enhances stability and the performance of models.

# Pipeline-based preprocessing

- A ColumnTransformer in a Scikit-learn pipeline does all transformations.

- This creates uniformity between inference and training.



# How to Run

1. Install dependencies:
   pip install streamlit scikit-learn shap lime pandas matplotlib

2. Run the app:
   streamlit run app.py

3. Upload the demo CSV file



# Model Description

In this project, we use a Random Forest classification model to predict whether a loan application should be approved or rejected.

The working process of the model.

1- Input data
- The model gets the details of the applicant including income, education, credit history and loan amount.

2- Preprocessing

- StandardScaler is used to scale the numerical values.

3- Model training

- Random Forest Classifier is a learner that is trained on the processed data.

- It picks trends that are accepted and rejected loans.

4- Prediction

- In the case of a new applicant, the model predicts:

Loan Approved (Good Credit) or

Loan Rejected (Bad Credit)

- It also gives out probability scores of both results.

5- Explainability

- SHAP and LIME are used to explain the prediction, and the use of each feature to explain the decision is indicated.

## 🔍 SHAP Explanation

SHAP explains model predictions using game theory.

Formula:
f(x) = E[f(x)] + Σ φᵢ

Where:
- f(x): model prediction
- E[f(x)]: expected (average) prediction
- φᵢ: contribution of feature i

# Global SHAP
Shows overall feature importance across all samples.

# Local SHAP
Explains why a specific applicant was approved or rejected.

Positive values increase approval probability, negative values decrease it.


# LIME Explanation

LIME creates a local interpretable model around a single prediction.

Steps:
1. Generate perturbed samples near the selected data point
2. Get model predictions
3. Fit a simple interpretable model
4. Show feature contributions

LIME helps understand local decision behavior.


# Features Implemented

✔ Upload CSV file  
✔ Prediction with probabilities  
✔ Global SHAP explanation  
✔ Local SHAP explanation  
✔ LIME explanation  
✔ Clean feature names  
✔ Interactive Streamlit UI  

