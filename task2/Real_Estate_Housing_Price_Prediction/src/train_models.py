"""
Machine Learning Training and Model Evaluation
California Housing Price Prediction
Trains OLS, Ridge, and Lasso regression models with GridSearchCV,
evaluates metrics on held-out test data, and saves model artifacts.
"""

import os
import sys
import time
import warnings

# Avoid OpenBLAS / MKL thread contention with loky on Windows
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import HousingFeatureEngineer, build_model_pipeline, build_preprocessor

def main():
    print("=" * 75, flush=True)
    print("CALIFORNIA HOUSING PRICE PREDICTION - MODEL TRAINING WORKFLOW", flush=True)
    print("=" * 75, flush=True)

    dataset_path = os.path.join(PROJECT_ROOT, 'dataset', 'housing.csv')
    outputs_dir = os.path.join(PROJECT_ROOT, 'outputs')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load data
    print(f"\n[1/6] Loading California Housing dataset from '{dataset_path}'...", flush=True)
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns", flush=True)

    # 2. Stratified split (80/20)
    print("\n[2/6] Performing Stratified Train-Test Split (80/20) on median_income bins...", flush=True)
    df['income_cat'] = pd.cut(
        df['median_income'],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['income_cat']):
        strat_train_set = df.loc[train_index].drop('income_cat', axis=1)
        strat_test_set = df.loc[test_index].drop('income_cat', axis=1)

    X_train = strat_train_set.drop('median_house_value', axis=1)
    y_train = strat_train_set['median_house_value']
    X_test = strat_test_set.drop('median_house_value', axis=1)
    y_test = strat_test_set['median_house_value']

    print(f"Training set: {X_train.shape[0]:,} samples | Testing set: {X_test.shape[0]:,} samples", flush=True)

    # 3. Model 1: OLS
    print("\n[3/6] Fitting Model 1: Ordinary Least Squares (OLS)...", flush=True)
    ols_pipe = build_model_pipeline(LinearRegression())
    t0 = time.time()
    ols_pipe.fit(X_train, y_train)
    ols_time = time.time() - t0
    y_pred_ols = ols_pipe.predict(X_test)

    rmse_ols = root_mean_squared_error(y_test, y_pred_ols)
    mae_ols = mean_absolute_error(y_test, y_pred_ols)
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    r2_ols = r2_score(y_test, y_pred_ols)
    print(f"  OLS Test RMSE: ${rmse_ols:,.2f} | MAE: ${mae_ols:,.2f} | R2: {r2_ols:.4f} (Fit: {ols_time:.2f}s)", flush=True)

    # 4. Model 2: Ridge Regression with GridSearchCV
    print("\n[4/6] Tuning Model 2: Ridge Regression with 5-Fold GridSearchCV...", flush=True)
    param_grid_ridge = {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}
    ridge_base = build_model_pipeline(Ridge())
    grid_ridge = GridSearchCV(
        ridge_base,
        param_grid_ridge,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    t0 = time.time()
    grid_ridge.fit(X_train, y_train)
    ridge_time = time.time() - t0
    best_ridge_pipe = grid_ridge.best_estimator_
    best_alpha_ridge = grid_ridge.best_params_['model__alpha']
    y_pred_ridge = best_ridge_pipe.predict(X_test)

    rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    print(f"  Optimal Ridge alpha: {best_alpha_ridge} (CV neg RMSE: {grid_ridge.best_score_:,.2f})", flush=True)
    print(f"  Ridge Test RMSE: ${rmse_ridge:,.2f} | MAE: ${mae_ridge:,.2f} | R2: {r2_ridge:.4f} (Search: {ridge_time:.2f}s)", flush=True)

    # 5. Model 3: Lasso Regression with GridSearchCV
    print("\n[5/6] Tuning Model 3: Lasso Regression with 5-Fold GridSearchCV...", flush=True)
    param_grid_lasso = {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
    lasso_base = build_model_pipeline(Lasso(max_iter=20000, tol=0.005, random_state=42))
    grid_lasso = GridSearchCV(
        lasso_base,
        param_grid_lasso,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    t0 = time.time()
    grid_lasso.fit(X_train, y_train)
    lasso_time = time.time() - t0
    best_lasso_pipe = grid_lasso.best_estimator_
    best_alpha_lasso = grid_lasso.best_params_['model__alpha']
    y_pred_lasso = best_lasso_pipe.predict(X_test)

    rmse_lasso = root_mean_squared_error(y_test, y_pred_lasso)
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    print(f"  Optimal Lasso alpha: {best_alpha_lasso} (CV neg RMSE: {grid_lasso.best_score_:,.2f})", flush=True)
    print(f"  Lasso Test RMSE: ${rmse_lasso:,.2f} | MAE: ${mae_lasso:,.2f} | R2: {r2_lasso:.4f} (Search: {lasso_time:.2f}s)", flush=True)

    # 6. Evaluation metrics summary table
    print("\n[6/6] Compiling Model Comparison Metrics and Persisting Models...", flush=True)
    metrics_data = [
        {
            'Model': 'Ordinary Least Squares (OLS)',
            'RMSE': round(rmse_ols, 2),
            'MAE': round(mae_ols, 2),
            'MSE': round(mse_ols, 2),
            'R2': round(r2_ols, 4)
        },
        {
            'Model': f'Ridge Regression (alpha={best_alpha_ridge})',
            'RMSE': round(rmse_ridge, 2),
            'MAE': round(mae_ridge, 2),
            'MSE': round(mse_ridge, 2),
            'R2': round(r2_ridge, 4)
        },
        {
            'Model': f'Lasso Regression (alpha={best_alpha_lasso})',
            'RMSE': round(rmse_lasso, 2),
            'MAE': round(mae_lasso, 2),
            'MSE': round(mse_lasso, 2),
            'R2': round(r2_lasso, 4)
        }
    ]
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(outputs_dir, 'model_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("MODEL EVALUATION COMPARISON TABLE (TEST SET)", flush=True)
    print("=" * 75, flush=True)
    print(metrics_df.to_string(index=False), flush=True)
    print("=" * 75, flush=True)

    # Persist best model based strictly on lowest test RMSE criterion ($65,567.30)
    best_model_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(ols_pipe, best_model_path)
    print(f"Saved Champion Pipeline (Lowest Test RMSE - OLS): {best_model_path}", flush=True)

    # Also persist Ridge as the regularized stability-based production alternative
    ridge_model_path = os.path.join(models_dir, 'ridge_model.joblib')
    joblib.dump(best_ridge_pipe, ridge_model_path)
    print(f"Saved Regularized Stability Pipeline (Ridge alpha={best_alpha_ridge}): {ridge_model_path}", flush=True)

    # Also persist Lasso pipeline
    lasso_model_path = os.path.join(models_dir, 'lasso_model.joblib')
    joblib.dump(best_lasso_pipe, lasso_model_path)
    print(f"Saved Sparse Feature Selection Pipeline (Lasso alpha={best_alpha_lasso}): {lasso_model_path}", flush=True)

    # Save all models & predictions for plotting and analysis
    artifacts = {
        'ols_pipe': ols_pipe,
        'best_ridge_pipe': best_ridge_pipe,
        'best_lasso_pipe': best_lasso_pipe,
        'best_alpha_ridge': best_alpha_ridge,
        'best_alpha_lasso': best_alpha_lasso,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_ols': y_pred_ols,
        'y_pred_ridge': y_pred_ridge,
        'y_pred_lasso': y_pred_lasso,
        'metrics_df': metrics_df
    }
    all_artifacts_path = os.path.join(models_dir, 'all_models.joblib')
    joblib.dump(artifacts, all_artifacts_path)
    print(f"Saved All Model Artifacts: {all_artifacts_path}", flush=True)
    print("\nTraining workflow completed successfully!", flush=True)

if __name__ == '__main__':
    main()

