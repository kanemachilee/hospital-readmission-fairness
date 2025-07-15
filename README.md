# Predicting 30-Day Hospital Readmissions: Fairness Audit and Model Evaluation

This project analyzes a real-world healthcare dataset to predict 30-day hospital readmissions for diabetic patients. In addition to model training and evaluation, the project includes a fairness audit to assess potential disparities in performance across race and gender groups.

## Project Objectives

- Build a logistic regression model to classify whether a patient will be readmitted within 30 days
- Evaluate model performance using standard metrics (accuracy, recall, confusion matrix)
- Conduct a fairness audit by subgroup to evaluate false negative rates and recall
- Visualize disparities across demographic groups

## Dataset

- Source: [Kaggle Diabetes Readmission Dataset](https://www.kaggle.com/datasets/whenamancodes/diabetes-patient-readmission-prediction)
- Size: 101,000+ patient records
- Features include demographic data, hospital admission info, diagnoses, procedures, and medications

## Methods

- Language: Python
- Libraries: pandas, scikit-learn, seaborn, matplotlib
- Preprocessing: One-hot encoding, removal of high-cardinality and missing columns
- Model: Logistic Regression (sklearn), trained with 80/20 split

## Results Summary

- Accuracy: ~89%
- Recall on positive readmissions (within 30 days): ~0.02
- Model consistently under-predicts positive cases (bias toward “No Readmission”)
- False negative rates are high across all race groups
- Slight variation in recall by gender

## Key Takeaways

- High accuracy does not imply clinical usefulness when recall is low
- False negatives in hospital readmission prediction may carry serious patient risk
- Subgroup analysis is essential for detecting disparities in healthcare AI models
- Additional tuning or rebalancing may be needed to improve recall and fairness

## Repository Structure
hospital-readmission-fairness/
├── data/ # Raw dataset
├── plots/ # Visualizations
├── 01_exploration.ipynb # Data exploration and EDA
├── 02_cleaning.ipynb # Preprocessing and model training
├── 03_fairness_audit.ipynb # Fairness metrics and visualizations
└── README.md
