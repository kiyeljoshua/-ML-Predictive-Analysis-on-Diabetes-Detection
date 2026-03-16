# 🩺 Predictive Analysis on Diabetes Detection: A Machine Learning Project

> A machine learning study that predicts diabetes risk using patient health data — built with Python, evaluated across 6 classification algorithms, and achieving over 75% accuracy on unseen data.

---

## 📌 Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Problem](#the-problem)
3. [Dataset Overview](#dataset-overview)
4. [Project Workflow](#project-workflow)
5. [Key Findings & Insights](#key-findings--insights)
6. [Model Results](#model-results)
7. [Tools & Technologies](#tools--technologies)

---

## Executive Summary

Diabetes is a growing global health concern, and early detection is critical to preventing complications. This project applies machine learning to predict whether a patient is likely to have diabetes based on 8 clinical health indicators — including glucose level, BMI, age, and blood pressure.

We trained and compared **6 classification algorithms**, using a structured pipeline that includes data cleaning, exploratory analysis, hyperparameter tuning, and cross-validated model evaluation. The **Support Vector Classifier (SVC)** emerged as the best-performing model, achieving the highest combined score across accuracy, ROC AUC, and F1 metrics.

**Bottom line:** Glucose level is the single strongest predictor of diabetes in this dataset, followed by BMI and Age. The final model correctly identifies diabetic patients with over 80% test accuracy and a ROC AUC of 0.854.

---

## The Problem

Manual clinical screening is time-intensive and not always accessible. Can we use a patient's routine health data to reliably flag diabetes risk — before it's too late?

This project answers that question by building a predictive model that:
- Identifies the most important risk factors for diabetes
- Compares multiple machine learning approaches objectively
- Produces a generalizable model that performs well on new, unseen data

---

## Dataset Overview

| Property | Detail |
|---|---|
| **Source** | Pima Indians Diabetes Dataset |
| **Records** | 308 patients |
| **Features** | 8 clinical inputs + 1 target |
| **Target Variable** | `Outcome` — 0 (No Diabetes), 1 (Diabetes) |
| **Class Distribution** | 215 non-diabetic (69.8%) / 93 diabetic (30.2%) |

### Features Used

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight in kg / height in m²) |
| `DiabetesPedigreeFunction` | Genetic diabetes likelihood based on family history |
| `Age` | Age in years |

---

## Project Workflow

```
Raw Data  →  Preprocessing  →  EDA  →  Model Training  →  Evaluation  →  Best Model
```

### 1. Data Preprocessing
- Identified biologically impossible zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`
- Replaced zeros with `NaN` and imputed using **column medians** (via `SimpleImputer`)
- Result: A complete dataset of 308 records with no missing values

### 2. Exploratory Data Analysis (EDA)
- **Scatterplot** — Glucose vs. BMI colored by outcome; fit quadratic regression lines per group
- **Pairplot** — All feature combinations visualized, highlighting separation by diabetes outcome
- **Correlation Matrix (Heatmap)** — Linear relationships between all features and the outcome

### 3. Train-Validation-Test Split
Data split using **stratified sampling** to preserve class balance:

| Set | Size | Purpose |
|---|---|---|
| Training | 70% (215 records) | Teach the model |
| Validation | 20% (62 records) | Tune and compare models |
| Test | 10% (31 records) | Final blind evaluation |

### 4. Model Development & Tuning
- 6 models trained: KNN, Logistic Regression, SVC, Random Forest, Gradient Boosting, XGBoost
- Automated pipeline with **StandardScaler** + **GridSearchCV** (5-fold Stratified CV)
- Primary tuning metric: **ROC AUC**

### 5. Model Evaluation
Each model evaluated on three sets using:
- **Accuracy** — Overall correct predictions
- **ROC AUC** — Ability to separate diabetic vs. non-diabetic cases
- **F1-Score** — Balance between precision and recall
- **Combined Score** — Weighted composite: 50% Accuracy + 30% ROC AUC + 20% F1

---

## Key Findings & Insights

### 🔬 From the Correlation Matrix

| Feature | Correlation with Outcome | Interpretation |
|---|---|---|
| **Glucose** | **0.51** | Strongest predictor — higher glucose = higher diabetes risk |
| **BMI** | 0.28 | 2nd most important factor |
| **Age** | 0.26 | Older patients face higher risk |
| Pregnancies | 0.21 | Moderate association |
| Insulin | 0.22 | Moderate, with high spread |
| BloodPressure | 0.18 | Weak — less direct linear relationship |
| DiabetesPedigreeFunction | 0.15 | Weakest among all features |

> **Key takeaway:** Glucose alone explains more variance in diabetes outcome than any other variable. BMI and Age are meaningful secondary indicators, but no single feature is sufficient for accurate prediction — a multi-feature model is necessary.

### 📊 From the Scatter & Pair Plots
- Diabetic patients (Outcome = 1) cluster at **higher Glucose AND higher BMI** simultaneously
- The quadratic regression line for diabetic patients rises more steeply — suggesting a **non-linear compounding effect** between glucose and BMI
- Significant overlap between groups confirms that clinical features alone cannot perfectly separate cases — hence the need for ML

---

## Model Results

### Performance on Test Set

| Model | Test Accuracy | Test ROC AUC | Test F1 | Combined Score |
|---|---|---|---|---|
| **SVC** ⭐ | **80.65%** | **0.854** | **0.667** | **Best** |
| XGBoost | 74.19% | 0.813 | 0.556 | 2nd |
| Gradient Boosting | 74.19% | 0.788 | 0.556 | 3rd |
| Logistic Regression | 74.19% | 0.783 | 0.556 | 4th |
| Random Forest | 74.19% | 0.783 | 0.556 | 5th |
| KNN | 70.97% | 0.750 | 0.526 | 6th |

### 🏆 Best Model: Support Vector Classifier (SVC)

- **Test Accuracy: 80.65%**
- **Test ROC AUC: 0.854** — strongest ability to distinguish diabetic vs. non-diabetic
- **Consistent across all 3 sets** — no overfitting, unlike tree-based models
- **Top Feature across all models: Glucose** — confirming EDA findings

> Tree-based models (XGBoost, Gradient Boosting, Random Forest) showed high training accuracy (>90%) but dropped significantly on the test set — a sign of overfitting. SVC and Logistic Regression generalized better to unseen data.

### Scoring Methodology
Combined Score = (50% × Accuracy) + (30% × ROC AUC) + (20% × F1)

Each metric internally weighted: 10% Train + 40% Validation + 50% Test — prioritizing real-world generalization over training performance.

---

## Tools & Technologies

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Models Used** | KNN, Logistic Regression, SVC, Random Forest, Gradient Boosting, XGBoost |
| **Tuning** | GridSearchCV, StratifiedKFold (5-fold CV) |
| **Preprocessing** | SimpleImputer (median), StandardScaler |
| **Environment** | Google Colab / Jupyter Notebook |

---



## License

This project is for academic and portfolio purposes.


