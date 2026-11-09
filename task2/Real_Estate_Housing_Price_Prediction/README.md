# Real Estate Housing Price Prediction using Ridge and Lasso Regression

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Machine Learning system developed to predict median house values across California census block groups. This project implements and rigorously compares **Ordinary Least Squares (OLS)**, **Ridge Regression ($L_2$ Regularization)**, and **Lasso Regression ($L_1$ Regularization)** to resolve severe multicollinearity, eliminate data leakage, and deliver reliable, interpretable real estate valuations.

---

## Table of Contents
1. [Project Introduction](#project-introduction)
2. [Problem Statement](#problem-statement)
3. [Project Objectives](#project-objectives)
4. [Dataset Description & Source](#dataset-description--source)
5. [Technologies & Tools](#technologies--tools)
6. [Project Folder Structure](#project-folder-structure)
7. [End-to-End Machine Learning Workflow](#end-to-end-machine-learning-workflow)
8. [Algorithms & Regularization Theory](#algorithms--regularization-theory)
9. [Multicollinearity Analysis (VIF)](#multicollinearity-analysis-vif)
10. [Evaluation Metrics & Actual Results](#evaluation-metrics--actual-results)
11. [Residual Error Analysis & Diagnostics](#residual-error-analysis--diagnostics)
12. [Installation Instructions](#installation-instructions)
13. [Execution Instructions](#execution-instructions)
14. [Sample Inference & Production Prediction](#sample-inference--production-prediction)
15. [Limitations & Future Enhancements](#limitations--future-enhancements)
16. [Conclusion](#conclusion)

---

## 1. Project Introduction

Accurate real estate property valuation is critical for institutional mortgage underwriters, property tax assessors, real estate developers, and individual homeowners. While traditional linear models are widely favored for their interpretability and computational speed, real-world geospatial and demographic datasets frequently violate classical OLS regression assumptions—most notably the **absence of multicollinearity**.

In this project, we implement a complete machine learning lifecycle using the 1990 California Housing census dataset. We design custom transformers and modular Scikit-Learn pipelines to engineer domain-specific per-household metrics and pairwise polynomial interactions. We evaluate Ordinary Least Squares against Ridge and Lasso regularized regression to understand how $L_2$ quadratic shrinkage and $L_1$ sparse feature selection stabilize coefficient variance in the presence of severe collinearity.

---

## 2. Problem Statement

Given a feature vector $\mathbf{x}_i \in \mathbb{R}^p$ representing the physical, demographic, and spatial characteristics of a California census block group, predict the continuous target variable $y_i \in \mathbb{R}$ (`median_house_value` in USD):

$$\hat{y}_i = \beta_0 + \sum_{j=1}^p x_{ij} \beta_j$$

### Challenges:
- **Severe Multicollinearity:** Counts of rooms, bedrooms, households, and population grow synchronously, inflating coefficient variance in unregularized OLS.
- **Data Leakage Risk:** Imputation medians, scaling factors, and categorical encoders must be fitted exclusively on training data and applied without leakage.
- **Target Value Truncation:** The target variable is artificially censored at **$500,001**, requiring careful residual diagnostics.

---

## 3. Project Objectives

- **Exploratory Data Analysis (EDA):** Deep dive into spatial distributions, target right-skewness, and pairwise correlations.
- **Safe Feature Engineering:** Engineer domain-specific ratios (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`) with safe division to avoid divide-by-zero errors.
- **Interaction Modeling:** Capture non-linear geographic and wealth interactions using `PolynomialFeatures(degree=2, interaction_only=True)`.
- **Multicollinearity Diagnostics:** Compute Variance Inflation Factors (VIF) using `statsmodels` to pinpoint ill-conditioned feature dimensions.
- **Hyperparameter Optimization:** Conduct 5-fold cross-validation (`GridSearchCV`) across logarithmic grids of $\alpha$ for Ridge and Lasso models.
- **Residual Diagnostics:** Investigate error normality (Q-Q plots), heteroscedasticity, and the $500,001 price ceiling artifact.
- **Production Deployment:** Persist the best complete pipeline (`joblib`) and provide a standalone production CLI inference tool (`src/predict.py`).

---

## 4. Dataset Description & Source

- **Source:** [Hands-On Machine Learning Repository (Aurélien Géron)](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)
- **Local Path:** `dataset/housing.csv`
- **Dimensions:** 20,640 records, 10 columns (9 features, 1 continuous target)

| Feature | Data Type | Missing Count | Description |
| :--- | :--- | :--- | :--- |
| `longitude` | float64 | 0 | Geographic coordinate (East-West) |
| `latitude` | float64 | 0 | Geographic coordinate (North-South) |
| `housing_median_age`| float64 | 0 | Median age of housing units in block group |
| `total_rooms` | float64 | 0 | Total room count across all homes |
| `total_bedrooms` | float64 | 207 (1.00%) | Total bedroom count (imputed via training median) |
| `population` | float64 | 0 | Total residents in block group |
| `households` | float64 | 0 | Total occupied housing units |
| `median_income` | float64 | 0 | Median household income in tens of thousands of USD |
| `ocean_proximity` | object | 0 | Categorical: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN` |
| **`median_house_value`** | float64 | 0 | **Target:** Median house value in USD |

---

## 5. Technologies & Tools

- **Programming Language:** Python 3.12
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn (`Pipeline`, `ColumnTransformer`, `GridSearchCV`, `StandardScaler`, `OneHotEncoder`, `PolynomialFeatures`, `SimpleImputer`)
- **Statistical Modeling & Diagnostics:** Statsmodels (VIF), SciPy (Probplot Q-Q)
- **Visualization:** Matplotlib, Seaborn
- **Serialization & Persistence:** Joblib
- **Interactive Documentation:** Jupyter Notebook (`nbformat`)

---

## 6. Project Folder Structure

```
Real_Estate_Housing_Price_Prediction/
│
├── dataset/
│   └── housing.csv                   # Validated raw California Housing dataset
│
├── notebooks/
│   └── housing_price_prediction.ipynb # Fully executed, documented 24-section notebook
│
├── outputs/                          # High-resolution (300 DPI) plots & metrics
│   ├── correlation_heatmap.png       # Pearson correlation matrix heatmap
│   ├── target_distribution.png       # Target variable histogram & boxplot
│   ├── feature_distributions.png     # Univariate feature distribution grid
│   ├── actual_vs_predicted_ols.png   # OLS actual vs predicted scatter plot
│   ├── actual_vs_predicted_ridge.png # Ridge actual vs predicted scatter plot
│   ├── actual_vs_predicted_lasso.png # Lasso actual vs predicted scatter plot
│   ├── residual_distribution.png     # Residual histogram, boxplot & Q-Q plot
│   ├── residual_vs_predicted.png     # Residuals vs fitted values scatter plot
│   ├── model_comparison.png          # Bar chart comparison (RMSE, MAE, R2)
│   ├── coefficient_comparison.png    # Horizontal coefficient comparison chart
│   └── model_metrics.csv             # Tabular model evaluation metrics
│
├── models/
│   └── best_model.joblib             # Serialized production pipeline
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── preprocessing.py              # Custom transformers & pipeline definitions
│   ├── build_project.py              # Master build & execution script
│   └── predict.py                    # Standalone CLI production inference script
│
├── requirements.txt                  # Complete pinned dependency list
├── README.md                         # Comprehensive project documentation
└── .gitignore                        # Python, Jupyter, OS, and cache ignore rules
```

---

## 7. End-to-End Machine Learning Workflow

```
[Raw Data: dataset/housing.csv]
               │
               ▼
[Stratified Train-Test Split (80/20)] ──> Based on median_income categories (seed=42)
               │
        ┌──────┴──────────────────────────────────────────────────────┐
        ▼                                                             ▼
[Training Partition: 16,512 rows]                             [Test Partition: 4,128 rows]
        │                                                             │ (Untouched until
        ▼                                                             │  final evaluation)
[Feature Engineering Transformer]                                     │
  - rooms_per_household = total_rooms / households                    │
  - bedrooms_per_room = total_bedrooms / total_rooms                  │
  - population_per_household = population / households                │
        │                                                             │
        ▼                                                             │
[ColumnTransformer Preprocessing Pipeline]                            │
  ├── Poly Pipeline: Imputer(median) ──> PolynomialFeatures ──> Scaler│
  ├── Num Pipeline:  Imputer(median) ──> StandardScaler               │
  └── Cat Pipeline:  Imputer(most_frequent) ──> OneHotEncoder         │
        │                                                             │
        ├──────────────────────┬──────────────────────┐               │
        ▼                      ▼                      ▼               │
[Model 1: OLS]         [Model 2: Ridge]       [Model 3: Lasso]        │
LinearRegression()      GridSearchCV (5-fold)  GridSearchCV (5-fold)  │
                        alphas=[0.01..1000]    alphas=[0.001..100]    │
        │                      │                      │               │
        └──────────────────────┼──────────────────────┘               │
                               ▼                                      │
                   [Model Selection & Calibration]                    │
                               │                                      │
                               ▼                                      ▼
                   [Final Evaluation on Held-Out Test Set] <──────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  [Residual Error Diagnostics]     [Production Model Serialization]
  - Q-Q plots, Histograms,          - models/best_model.joblib
  - Heteroscedasticity analysis     - CLI scoring via src/predict.py
```

---

## 8. Algorithms & Regularization Theory

### 1. Ordinary Least Squares (OLS)
OLS finds parameters $\boldsymbol{\hat{\beta}}$ by minimizing the sum of squared residuals:

$$\min_{\boldsymbol{\beta}} \text{RSS}(\boldsymbol{\beta}) = \min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$$

Closed-form solution:

$$\boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

When $\mathbf{X}$ exhibits multicollinearity, the matrix $\mathbf{X}^T \mathbf{X}$ has near-zero eigenvalues, causing parameter variance $\text{Var}(\boldsymbol{\hat{\beta}}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$ to explode.

### 2. Ridge Regression ($L_2$ Regularization)
Ridge adds a quadratic penalty proportional to the Euclidean norm of the coefficient vector:

$$\min_{\boldsymbol{\beta}} \left( \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_2^2 \right)$$

Closed-form solution:

$$\boldsymbol{\hat{\beta}}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

The addition of $\alpha \mathbf{I}$ strictly bounds the eigenvalues away from zero, guaranteeing a stable inverse and shrinking collinear coefficients proportionally.

### 3. Lasso Regression ($L_1$ Regularization)
Lasso adds an absolute-value penalty:

$$\min_{\boldsymbol{\beta}} \left( \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1 \right)$$

Due to the non-differentiable sharp corners of the $L_1$ diamond constraint sphere at zero, optimization via coordinate descent drives non-essential coefficients to **exactly zero**, performing automatic feature selection and producing sparse, highly interpretable models.

---

## 9. Multicollinearity Analysis (VIF)

Variance Inflation Factors (VIF) were calculated on standardized numerical predictors using `statsmodels`:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

| Feature | VIF Value | Multicollinearity Level | Action Taken |
| :--- | :--- | :--- | :--- |
| `total_bedrooms` | **36.00** | Severe ($\text{VIF} > 10$) | Regularized via Ridge/Lasso shrinkage |
| `households` | **35.14** | Severe ($\text{VIF} > 10$) | Regularized via Ridge/Lasso shrinkage |
| `total_rooms` | **12.72** | High ($\text{VIF} > 10$) | Regularized via Ridge/Lasso shrinkage |
| `latitude` | 8.83 | Moderate-High | Captured via spatial interaction term |
| `longitude` | 8.71 | Moderate-High | Captured via spatial interaction term |
| `population` | 6.37 | Moderate | Regularized via Ridge shrinkage |
| `median_income` | 1.73 | Low ($\text{VIF} < 5$) | Retained as primary linear driver |
| `housing_median_age`| 1.26 | Low ($\text{VIF} < 5$) | Retained cleanly |

---

## 10. Evaluation Metrics & Actual Results

All models were evaluated on the held-out 20% test partition ($n = 4,128$) using:
- **Root Mean Squared Error (RMSE):** Measures standard deviation of residuals (penalizes large outlier mistakes).
- **Mean Absolute Error (MAE):** Average magnitude of absolute errors (robust to outliers).
- **Mean Squared Error (MSE):** Variance of residuals.
- **Coefficient of Determination ($R^2$):** Proportion of variance explained by model.

### Actual Model Performance Comparison Table:

| Model | Hyperparameter ($\alpha$) | Test RMSE ($) | Test MAE ($) | Test MSE | Test $R^2$ Score | Designation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ordinary Least Squares (OLS)** | None | **$65,567.30** | **$47,424.63** | **$4.30 \times 10^9$** | **0.6701** | **Lowest-RMSE Champion (Saved as `best_model.joblib`)** |
| **Ridge Regression** | $\alpha = 0.01$ | $65,573.70 | $47,439.76 | $4.30 \times 10^9$ | **0.6701** | **Regularized Stability Model (Saved as `ridge_model.joblib`)** |
| **Lasso Regression** | $\alpha = 0.001$ | $65,878.29 | $47,840.12 | $4.34 \times 10^9$ | **0.6670** | Sparse Feature Selection Alternative |

*Note: All values reflect actual computed test metrics from `outputs/model_metrics.csv`.*

### Model Selection & Architectural Rationale:
1. **Lowest-RMSE Champion Model (OLS)**:
   - Under the task's explicit evaluation criterion (lowest test RMSE), **Ordinary Least Squares (OLS)** is selected as the best-performing model, achieving a test RMSE of **$65,567.30**, a test MAE of **$47,424.63**, and an $R^2$ of **0.6701**.
   - The complete fitted OLS pipeline (including median imputation, feature engineering, polynomial interactions, standard scaling, and one-hot encoding) is serialized as the primary model to **`models/best_model.joblib`**.
2. **Regularized Stability Alternative (Ridge Regression)**:
   - While OLS achieves the lowest test error by a margin of $6.40 (a 0.009% difference), our VIF analysis diagnosed severe multicollinearity ($\text{VIF} > 35$) between `total_bedrooms` and `households`.
   - **Ridge Regression ($\alpha = 0.01$)** adds an $L_2$ penalty ($\alpha \mathbf{I}$), conditioning the matrix inverse and dampening coefficient variance. Ridge is persisted to **`models/ridge_model.joblib`** as a production-grade alternative for operational environments where input data may shift.

---

## 11. Residual Error Analysis & Diagnostics

1. **Mean Residual:** The mean residual for OLS is near zero (**-$8.15**), and for Ridge is **$144.38** (standard deviation: $65,567.30), demonstrating an unbiased regression fit centered on zero.
2. **Normality (Q-Q Plot):** Quantile-Quantile analysis reveals that errors follow a normal distribution within $\pm 1.5$ standard deviations, but display heavy tails at both extremes.
3. **The $500,001 Truncation Ceiling:** Exactly **965 properties (4.68% of the dataset)** are capped strictly at $500,001 (with 992 properties / 4.81% at or above $500,000). For these units, actual value cannot exceed $500,001 even when true market value is $800,000+. This generates a visible diagonal line in the residual plot ($e_i = 500,001 - \hat{y}_i$) with slope $-1$.
4. **Heteroscedasticity:** Residual scatter slightly widens for properties predicted above $350,000, which is standard in real estate as high-value homes exhibit higher absolute dollar variance.

---

## 12. Installation Instructions

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Git

### Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd Real_Estate_Housing_Price_Prediction

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 13. Execution Instructions

### 1. Run Complete Automated Workflow
To retrain models, generate all 10 visualizations, produce `model_metrics.csv`, and save the best pipeline:
```bash
python src/build_project.py
```

### 2. Run Jupyter Notebook
To inspect the interactive, fully documented notebook:
```bash
jupyter notebook notebooks/housing_price_prediction.ipynb
```

---

## 14. Sample Inference & Production Prediction

The standalone prediction script `src/predict.py` accepts raw housing features, validates boundaries, and displays formatted USD valuations.

### Benchmark Sample Predictions
```bash
python src/predict.py --sample
```

**Output (Default Champion OLS Pipeline):**
```
======================================================================
CALIFORNIA HOUSING PRICE PREDICTION - SAMPLE INFERENCE BENCHMARK
Active Model: OLS Linear Regression (Lowest-RMSE Champion)
======================================================================

Scenario: San Francisco Bay Area Coastal Home (High Income)
--------------------------------------------------
  longitude                : -122.25
  latitude                 : 37.85
  housing_median_age       : 42.0
  total_rooms              : 2500.0
  total_bedrooms           : 450.0
  population               : 1100.0
  households               : 420.0
  median_income            : 8.3252
  ocean_proximity          : NEAR BAY
  >>> Predicted Median House Value: $426,926.06

Scenario: Central Valley Inland Suburban Home (Moderate Income)
--------------------------------------------------
  longitude                : -119.8
  latitude                 : 36.75
  housing_median_age       : 25.0
  total_rooms              : 1800.0
  total_bedrooms           : 350.0
  population               : 1200.0
  households               : 340.0
  median_income            : 3.125
  ocean_proximity          : INLAND
  >>> Predicted Median House Value: $114,236.24

Scenario: Southern California Coastal Property (<1H OCEAN)
--------------------------------------------------
  longitude                : -118.3
  latitude                 : 34.05
  housing_median_age       : 35.0
  total_rooms              : 3200.0
  total_bedrooms           : 600.0
  population               : 1500.0
  households               : 580.0
  median_income            : 5.45
  ocean_proximity          : <1H OCEAN
  >>> Predicted Median House Value: $304,173.45
======================================================================
```
*Note: The regularized Ridge pipeline (`--model ridge`) predicts $427,115.98, $114,212.74, and $304,220.86 respectively, differing by less than $190.*

### Single Prediction via CLI Arguments
```bash
python src/predict.py --longitude -122.23 --latitude 37.88 --housing_median_age 41 --total_rooms 880 --total_bedrooms 129 --population 322 --households 126 --median_income 8.3252 --ocean_proximity "NEAR BAY"
```

### Single Prediction via JSON File / String
```bash
python src/predict.py --json '{"longitude": -122.25, "latitude": 37.85, "housing_median_age": 42.0, "total_rooms": 2500.0, "total_bedrooms": 450.0, "population": 1100.0, "households": 420.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}'
```

---

## 15. Limitations & Future Enhancements

1. **Upper Censoring ($500,001 Cap):** Exactly **965 records (4.68%)** are censored at $500,001 (and 992 records / 4.81% are $\ge \$500,000$). Future iterations can implement **Tobit Censored Regression** or hurdle models to handle censored bounds without downward bias.
2. **Temporal Stagnation:** The dataset reflects 1990 census figures. Real estate markets have shifted; inflation adjustments or recent MLS/census integrations would modernize absolute price levels.
3. **Non-Linear Geospatial Clustered Effects:** Linear regression struggles with hyper-localized price boundaries (e.g., zip-code lines or school districts). Combining Ridge/Lasso feature engineering with **Gradient Boosted Trees (XGBoost / LightGBM)** could push $R^2$ above 0.82.

---

## 16. Conclusion

This project successfully constructs an end-to-end, production-ready Machine Learning system that:
1. Identifies and quantifies extreme multicollinearity ($\text{VIF} > 35$) among census physical counts.
2. Employs Scikit-Learn Pipelines to completely eliminate data leakage during feature engineering, scaling, and categorical encoding.
3. Optimizes regularization parameters ($\alpha$) via 5-fold cross-validation.
4. Identifies **Ordinary Least Squares (OLS)** as the champion model under the task's lowest-RMSE metric ($\text{RMSE} = \$65,567.30$, $R^2 = 0.6701$), while providing **Ridge Regression** ($\text{RMSE} = \$65,573.70$, $R^2 = 0.6701$) as the regularized stability alternative.
5. Deploys the complete pipeline into a self-contained, validated production inference tool (`src/predict.py`).

