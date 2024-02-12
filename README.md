# Breast Cancer Diagnostic Analysis

This project aims to leverage various machine learning algorithms to accurately diagnose breast cancer based on features derived from digital images of fine needle aspirates of breast masses. We explore and compare the performance of several classifiers on the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Overview

Breast cancer is one of the most common cancers among women worldwide. Early and accurate diagnosis is crucial for effective treatment planning. This project utilizes machine learning to predict whether a tumor is benign or malignant, based on features extracted from biopsy images.

## Dataset

The dataset used in this analysis is the Breast Cancer Wisconsin (Diagnostic) Dataset, which includes measurements for 569 instances of breast masses, each with 30 features computed from a digitized image. The target variable indicates the diagnosis (malignant or benign).

## Models Evaluated

- Logistic Regression
- Support Vector Machines (SVM)
- Decision Trees
- Random Forests
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- Perceptron
- Multi-layer Perceptron classifier

## Methodology

The analysis follows these steps:
1. **Data Preprocessing**: Scaling of features and splitting the dataset into training and testing sets.
2. **Exploratory Data Analysis (EDA)**: Visual and statistical examination of the features to understand their distributions and relationships.
3. **Feature Selection**: Identification of the most informative features based on statistical tests and their effect sizes.
4. **Model Training and Evaluation**: Training of multiple machine learning models and evaluation of their performance using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
5. **Model Comparison**: Comparison of the models' performances to identify the most effective classifier for this task.

## Results

The analysis revealed that certain models, particularly the Multi-layer Perceptron and Logistic Regression, showed superior performance across multiple evaluation metrics, demonstrating their potential utility in assisting medical professionals with breast cancer diagnosis.

## Conclusion

Machine learning models, especially those with high precision and recall, can significantly aid in the early detection and diagnosis of breast cancer. Future work may explore deeper neural network architectures and more advanced ensemble methods to further improve diagnostic accuracy.

## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
