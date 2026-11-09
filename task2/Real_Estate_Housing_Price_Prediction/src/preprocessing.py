"""
Data Preprocessing and Feature Engineering Pipelines
California Housing Price Prediction
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso

class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for engineering California Housing features.
    
    Creates:
    - rooms_per_household = total_rooms / households
    - bedrooms_per_room = total_bedrooms / total_rooms
    - population_per_household = population / households
    
    Handles safe division to avoid division-by-zero errors.
    """
    def __init__(self, eps=1e-8):
        self.eps = eps
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            feature_cols = [
                'longitude', 'latitude', 'housing_median_age',
                'total_rooms', 'total_bedrooms', 'population',
                'households', 'median_income', 'ocean_proximity'
            ]
            X_df = pd.DataFrame(X, columns=feature_cols)
            
        households = X_df['households'].astype(float).replace(0, np.nan)
        total_rooms = X_df['total_rooms'].astype(float).replace(0, np.nan)
        
        X_df['rooms_per_household'] = X_df['total_rooms'] / households
        X_df['bedrooms_per_room'] = X_df['total_bedrooms'] / total_rooms
        X_df['population_per_household'] = X_df['population'] / households
        
        return X_df

def build_preprocessor():
    """
    Builds the ColumnTransformer preprocessor handling imputation,
    polynomial interaction features, standard scaling, and categorical encoding.
    """
    poly_cols = ['median_income', 'housing_median_age', 'rooms_per_household', 'latitude', 'longitude']
    remaining_num_cols = ['total_rooms', 'total_bedrooms', 'population', 'households', 'bedrooms_per_room', 'population_per_household']
    cat_cols = ['ocean_proximity']

    poly_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('scaler', StandardScaler())
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('poly', poly_pipeline, poly_cols),
        ('num', num_pipeline, remaining_num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    return preprocessor

def build_model_pipeline(model):
    """
    Constructs a complete end-to-end pipeline from raw DataFrame input to predictions.
    """
    return Pipeline([
        ('fe', HousingFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('model', model)
    ])

