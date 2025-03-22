# Breast Cancer Detection Model - Logistic Regression

## Overview
This project aims to develop a machine learning model for detecting breast cancer using historical medical data. The model uses the breast cancer dataset provided by Scikit-learn, and the classification task is handled using the **Logistic Regression** algorithm. The model achieved a validation accuracy of **92%** in predicting whether tumors are malignant or benign.

## Objective
The goal of this project was to build a reliable machine learning model for breast cancer detection using the Logistic Regression algorithm. This project aims to assist in early detection, which can significantly impact treatment and outcomes for patients.

## Dataset
- The dataset used is the **Breast Cancer Wisconsin (Diagnostic) dataset**, which is a built-in dataset in **Scikit-learn**.
- The dataset contains 30 features that describe characteristics of cell nuclei present in breast cancer biopsies. These features include:
  - Radius, texture, smoothness, compactness, symmetry, and more.
- The task is to predict whether the tumor is **malignant** (1) or **benign** (0).

## Algorithm Used
- **Logistic Regression:** A statistical model used for binary classification problems. Logistic Regression is a linear model for classification that predicts the probability of a binary outcome (in this case, benign or malignant tumors).

## Model Development

### 1. Data Loading and Preprocessing
- The Breast Cancer dataset is loaded using Scikit-learn's `load_breast_cancer()` method.
- The data is split into training and testing sets (80% for training, 20% for testing).
- The features are standardized using `StandardScaler` to ensure all features contribute equally to the model.

### 2. Model Training
- The **Logistic Regression** model is trained using the training dataset.
- Default parameters are used for Logistic Regression, with the maximum iterations set to 10000 to ensure convergence.

### 3. Model Evaluation
- The model is evaluated on the test dataset using **accuracy** as the primary metric.
- The model achieved a **92% validation accuracy**.

## Results
The Logistic Regression model demonstrated **92% accuracy** on the validation set, making it a reliable tool for predicting breast cancer outcomes. The model classifies tumors into **malignant** or **benign** categories.

## Requirements

- Python 3.x
- Scikit-learn
- NumPy
- Pandas
- Matplotlib (for visualization)
- Jupyter Notebook (for development)

#
To use the model, follow these steps:

1. Load and preprocess the dataset.
2. Train the Logistic Regression model.
3. Evaluate the model on the test set to get the accuracy.


