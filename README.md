# Blockchain Fraud Detection — Comparative ML Analysis

## Overview
This project performs a comparative analysis of five machine learning algorithms to detect illicit blockchain transactions.  
**Primary challenge:** illicit transactions were under 10% of the dataset — the major task was ensuring the minority (illicit) class was included in training and properly evaluated. This was handled using **SMOTE** and class-aware evaluation (precision / recall / F1 for the illicit class). The original notebook was created and executed on **Kaggle**.

## Highlights
- Models compared: **Logistic Regression**, **Naïve Bayes**, **MLP (Neural Network)**, **Decision Tree**, **XGBoost**.
- Imbalance handling: **SMOTE (Imbalanced-learn)** applied to training folds with care to avoid data leakage.
- Hyperparameter tuning: **GridSearchCV** used for **MLP** and **XGBoost** (tuned versions reported below).
- Evaluation metrics focused on the **illicit class**: Precision (Illicit), Recall (Illicit), F1-Score (Illicit), ROC-AUC, and overall Accuracy.

## Reproducible Workflow
1. Data cleaning & feature engineering (missing values, scaling, encoding).  
2. Train/test split with stratification to keep the minority class distribution.  
3. Apply **SMOTE** on the training set only (no leakage into validation/test).  
4. Train baseline models; then tune MLP & XGBoost with `GridSearchCV` using `scoring='f1'` (illicit class).  
5. Evaluate on the held-out test set and report metrics for the illicit class.

## Results (Test set — Illicit-class focused)
| Model (Tuned if applicable) | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) |
|-----------------------------|----------:|--------------------:|-----------------:|-------------------:|
| **XGBoost (Tuned)**         | 99.10%   | 98.36%              | 92.30%           | **95.23%**         |
| **MLP (Tuned)**             | 97.59%   | 86.63%              | 89.11%           | 87.85%             |
| **Decision Tree**           | 97.04%   | 81.31%              | 90.43%           | 85.63%             |
| **Logistic Regression**     | 89.85%   | 48.95%              | 92.52%           | 64.03%             |
| **Naive Bayes**             | 90.24%   | 0.00%               | 0.00%            | 0.00%              |

> **Interpretation:** Tuned **XGBoost** produced the best F1 for the illicit class (95.23%), showing the strongest balance of precision and recall on the minority class after SMOTE + CV tuning. Naive Bayes failed to capture illicit samples (0 precision/recall), highlighting the importance of model choice and balancing.

## Notebooks / Files
- `blockchain-fraud-detection.ipynb` — full Kaggle notebook with step-by-step code, experiments, and plots.
- `requirements.txt` — environment packages.
- `README.md` — this file.

## How to run locally
```bash
git clone https://github.com/<your-username>/blockchain-fraud-detection.git
cd blockchain-fraud-detection
pip install -r requirements.txt
jupyter notebook blockchain-fraud-detection.ipynb
