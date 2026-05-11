# 💳 Credit Card Customer Segment Classification: Scale, Imbalance & Business Impact

## 🎯 Executive Summary
In the hyper-competitive financial sector, understanding customer behavior through precise segmentation is the key to maximizing marketing ROI and minimizing churn. This project delivers a high-performance classification pipeline to segment **2.4 million credit card customers** using a high-dimensional dataset of **857 features**.

## 🚀 Executive Summary (TL;DR)
- **The Problem**: Segmenting 2.4M credit card customers with 857 features, suffering from extreme class imbalance (minority class < 0.01%) and complex missing patterns.
- **The Solution**: Built a robust pipeline using **Dask** for big data scale, predictive ML imputation for missing data, and a **Stacking Ensemble** (CatBoost + LogReg + MLP).
- **The Result**: Achieved **Top 25% (58th Place)** in the competition with a validation F1-score of **0.8936**.

## 🛠 Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-EE4C2C?style=for-the-badge&logo=xgboost&logoColor=white)
![Dask](https://img.shields.io/badge/Dask-F7931E?style=for-the-badge&logo=dask&logoColor=white)

---

## 💡 1. Business & Data Understanding (비즈니스 및 데이터 이해)

### The Business Problem
A financial institution wants to identify customer segments (A to E) to deploy targeted marketing campaigns. Misclassification leads to wasted marketing budget or, worse, customer annoyance. The goal is to maximize the F1-score to ensure balanced precision and recall across all segments.

### The Data Challenge
- **Volume**: 2,400,000 rows × 857 columns.
- **Sparsity**: High rate of missing values requiring domain-specific handling.
- **Extreme Imbalance**: 
  - Class E (Majority): ~1.9M
  - Class B (Extreme Minority): 144
  - *Strategy*: Oversampling minority classes (A, B) and controlled undersampling of majority classes (C, D, E) to train a balanced model.

---

## 📊 2. Advanced Preprocessing & Feature Engineering (고급 전처리 및 피처 엔지니어링)

Instead of applying blind automation, this project implements a **domain-driven and predictive preprocessing pipeline**:

### 🧠 Predictive Imputation (머신러닝 기반 예측 대체)
Key features like `혜택수혜율_R3M` (Benefit Usage Rate) had high missing rates. Instead of filling them with static zeros or means:
- We trained a **Multi-Output RandomForest Regressor** on the non-missing data.
- Predicted the missing values based on customer age, gender, and current usage amounts.
- *Impact*: Preserved the variance and correlation structure of the data, leading to a more realistic feature space.

### 🛠️ Rule-Based Domain Logic
- **Telecom & Residence Sync**: Validated `가입통신회사코드` and filled missing `직장시도명` using `거주시도명` based on geographical probability.
- **Missingness as a Feature**: Created binary flags for missing patterns. The fact that a variable is missing is often a strong behavioral signal in credit data.

---

## 🤖 3. Modeling: Bridging Trees and Deep Learning (모델링 전략)

We benchmarked state-of-the-art tabular models to find the optimal balance between training speed and predictive power.

### Experiments & Results

| Model | Strategy | F1-Score | Key Insight |
| :--- | :--- | :---: | :--- |
| **XGBoost** | Baseline | 0.607 | Fast but struggled with extreme imbalance. |
| **CatBoost** | Full Feature Set | **0.8893 (Val)** | Best single model. Handled categorical features natively. |
| **TabNet** | Deep Learning | 0.8285 | Captured complex non-linearities but slower than trees. |
| **Stacking Ensemble** | CatBoost + LogReg + MLP | **0.8936 (Val)** | **Final Choice**. Combined the best of both worlds. |

### 📈 Model Performance Comparison
![Model F1 Score Comparison](images/model_f1_comparison.png)
*Figure: Comparison of Weighted F1-Scores across different models.*

---

## 🚀 4. How to Run & Reproduce (재현 방법)

### Prerequisites
```bash
pip install -r requirements.txt
```

### Execution Pipeline
1. **Data Analysis**: Run `src/missing_mechanism_analysis.py` to understand missing patterns.
2. **Preprocessing**: Run `src/preprocess_missing_features.py` to apply the predictive imputation.
3. **Training**: Run `src/train_eval_20k.py` to train the CatBoost and Stacking models.

---

## 🏁 5. Future Work & Commercial Expansion
- **SHAP (Explainable AI)**: Implement SHAP to explain *why* a customer is classified into a specific segment, providing actionable insights for the marketing team.
- **Cost-Sensitive Learning**: Implement custom loss functions to penalize misclassification of the rare but high-value segments (Classes A and B).

---

## 📁 Repository Structure
```text
├── notebooks/                  # Exploratory and experimental notebooks
│   ├── missing_mechanism_analysis.ipynb
│   ├── tabnet_experiments.ipynb
│   └── train_eval_20k.ipynb
├── src/                        # Extracted Python scripts from notebooks
│   ├── baseline_xgb.py
│   ├── missing_mechanism_analysis.py
│   ├── preprocess_missing_features.py
│   ├── preprocess_overview.py
│   ├── scaling_log_standard.py
│   ├── tabnet_experiments.py
│   └── train_eval_20k.py
├── reports/                    # Experiment summaries and reports
│   └── 실험요약.md
└── README.md                   # Project documentation
```

## 👥 Contributors
- **Junhyung L.** (Project Lead)

---
*Refactored and polished to meet professional software engineering standards for the [Data Analyst Portfolio](https://github.com/junhyung-L).*
*Note: Statistical findings and feature importances are based on the actual competition report results.*


