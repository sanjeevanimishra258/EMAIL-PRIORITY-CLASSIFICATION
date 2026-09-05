# EMAIL-PRIORITY-CLASSIFICATION
Binary classification using Logistic Regression to predict email priority
# Email Priority Classification

## Overview
This project develops a binary classification model using Logistic Regression to predict whether an incoming email should be marked as high priority based on sender history, message urgency, and metadata.

Completed for Track 2 Classification Practice, Problem 08.

## Objective
Train a binary classification model using the following predictors:
* `sender_frequency`
* `keyword_score`
* `thread_length`
* `response_deadline_hours`
* `attachment_count`
* `previous_priority_rate`

**Target:**
* `1` = High Priority
* `0` = Normal / Low Priority

## Dataset
* **File:** `dataset_08_email_priority_classification.csv`
* **Samples:** 1,000 rows, 7 columns (6 features + 1 target)
* **Integrity:** 0 missing values, 0 duplicate rows

## Technologies Used
* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib

## Machine Learning Workflow
1. Inspected dataset dimensions, integrity, and class distribution.
2. Separated input features ($X$) and target variable ($y$).
3. Stratified 80/20 train/test split to preserve class ratios without data leakage.
4. Feature standardization using `StandardScaler` fitted strictly on training data.
5. Model training using `LogisticRegression(max_iter=1000)`.
6. Performance evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
7. Generated Confusion Matrix (`Figure_1.png`) and ROC Curve (`Figure_2.png`).
8. Feature coefficient interpretation to assess priority drivers.

## Model Configuration
```python
LogisticRegression(max_iter=1000, random_state=42)
