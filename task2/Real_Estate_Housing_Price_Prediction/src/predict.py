"""
Production Inference Script for California Housing Price Prediction
Loads trained pipeline and predicts house prices from raw feature inputs.
"""

import os
import sys
import argparse
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import transformer so joblib unpickles correctly
try:
    from src.preprocessing import HousingFeatureEngineer
except ImportError:
    from preprocessing import HousingFeatureEngineer

# Register in __main__ to ensure deserialization compatibility
sys.modules['__main__'].HousingFeatureEngineer = HousingFeatureEngineer

FEATURE_NAMES = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income',
    'ocean_proximity'
]

VALID_OCEAN_PROXIMITIES = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']

VALIDATION_RANGES = {
    'longitude': (-125.0, -114.0),
    'latitude': (32.0, 43.0),
    'housing_median_age': (1.0, 100.0),
    'total_rooms': (1.0, 50000.0),
    'total_bedrooms': (1.0, 10000.0),
    'population': (1.0, 50000.0),
    'households': (1.0, 10000.0),
    'median_income': (0.1, 20.0)
}

def load_pipeline(model_choice='best'):
    """Loads the trained machine learning pipeline."""
    if model_choice in ['best', 'ols']:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib')
    elif model_choice == 'ridge':
        model_path = os.path.join(PROJECT_ROOT, 'models', 'ridge_model.joblib')
    elif model_choice == 'lasso':
        model_path = os.path.join(PROJECT_ROOT, 'models', 'lasso_model.joblib')
    elif os.path.exists(str(model_choice)):
        model_path = model_choice
    else:
        model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.joblib')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run the training pipeline first."
        )
        
    pipeline = joblib.load(model_path)
    return pipeline

def validate_input(data_dict):
    """
    Validates input features for correctness, data types, and logical boundaries.
    Returns cleaned dictionary or raises ValueError.
    """
    cleaned = {}
    for feature in FEATURE_NAMES:
        if feature not in data_dict:
            raise ValueError(f"Missing required feature: '{feature}'")
            
        val = data_dict[feature]
        
        if feature == 'ocean_proximity':
            val_str = str(val).strip().upper()
            if val_str not in VALID_OCEAN_PROXIMITIES:
                raise ValueError(
                    f"Invalid 'ocean_proximity': '{val}'. "
                    f"Must be one of {VALID_OCEAN_PROXIMITIES}"
                )
            cleaned[feature] = val_str
        else:
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feature}' must be a valid number, got '{val}'")
                
            min_val, max_val = VALIDATION_RANGES[feature]
            if not (min_val <= num_val <= max_val):
                raise ValueError(
                    f"Feature '{feature}' value {num_val} is out of realistic range "
                    f"[{min_val}, {max_val}]"
                )
            cleaned[feature] = num_val
            
    # Relational validation
    if cleaned['total_bedrooms'] > cleaned['total_rooms']:
        raise ValueError(
            f"total_bedrooms ({cleaned['total_bedrooms']}) cannot exceed total_rooms ({cleaned['total_rooms']})"
        )
        
    return cleaned

def predict_single(data_dict, model_path=None):
    """
    Predicts house value for a single observation dictionary.
    """
    cleaned = validate_input(data_dict)
    pipeline = load_pipeline(model_path)
    
    # Preserve exact feature names and order
    df_input = pd.DataFrame([cleaned])[FEATURE_NAMES]
    
    pred = pipeline.predict(df_input)[0]
    return float(pred)

def format_usd(amount):
    """Formats a float as USD currency."""
    return f"${amount:,.2f}"

def run_sample_predictions(model_choice='best'):
    """Runs prediction on diverse representative benchmark samples."""
    samples = [
        {
            "name": "San Francisco Bay Area Coastal Home (High Income)",
            "data": {
                "longitude": -122.25,
                "latitude": 37.85,
                "housing_median_age": 42.0,
                "total_rooms": 2500.0,
                "total_bedrooms": 450.0,
                "population": 1100.0,
                "households": 420.0,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }
        },
        {
            "name": "Central Valley Inland Suburban Home (Moderate Income)",
            "data": {
                "longitude": -119.80,
                "latitude": 36.75,
                "housing_median_age": 25.0,
                "total_rooms": 1800.0,
                "total_bedrooms": 350.0,
                "population": 1200.0,
                "households": 340.0,
                "median_income": 3.1250,
                "ocean_proximity": "INLAND"
            }
        },
        {
            "name": "Southern California Coastal Property (<1H OCEAN)",
            "data": {
                "longitude": -118.30,
                "latitude": 34.05,
                "housing_median_age": 35.0,
                "total_rooms": 3200.0,
                "total_bedrooms": 600.0,
                "population": 1500.0,
                "households": 580.0,
                "median_income": 5.4500,
                "ocean_proximity": "<1H OCEAN"
            }
        }
    ]
    
    pipeline = load_pipeline(model_choice)
    model_desc = "OLS Linear Regression (Lowest-RMSE Champion)" if model_choice in ['best', 'ols'] else f"Model ({model_choice})"
    print("=" * 70)
    print("CALIFORNIA HOUSING PRICE PREDICTION - SAMPLE INFERENCE BENCHMARK")
    print(f"Active Model: {model_desc}")
    print("=" * 70)
    
    for s in samples:
        print(f"\nScenario: {s['name']}")
        print("-" * 50)
        for k, v in s['data'].items():
            print(f"  {k:25s}: {v}")
            
        df_in = pd.DataFrame([s['data']])[FEATURE_NAMES]
        price = pipeline.predict(df_in)[0]
        print(f"  >>> Predicted Median House Value: {format_usd(price)}")
        
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description="Predict California Median Housing Prices using trained OLS, Ridge, or Lasso models."
    )
    parser.add_argument(
        "--model", type=str, default="best", choices=["best", "ols", "ridge", "lasso"],
        help="Model pipeline to load: 'best'/'ols' (lowest RMSE), 'ridge' (regularized stability), 'lasso' (sparse feature selection)."
    )
    parser.add_argument(
        "--sample", action="store_true", help="Run benchmark sample predictions."
    )
    parser.add_argument(
        "--json", type=str, help="JSON string or path to JSON file containing feature dictionary."
    )
    parser.add_argument("--longitude", type=float, help="Longitude (e.g., -122.25)")
    parser.add_argument("--latitude", type=float, help="Latitude (e.g., 37.85)")
    parser.add_argument("--housing_median_age", type=float, help="Median age of house in block (e.g., 42.0)")
    parser.add_argument("--total_rooms", type=float, help="Total rooms in block (e.g., 2500.0)")
    parser.add_argument("--total_bedrooms", type=float, help="Total bedrooms in block (e.g., 450.0)")
    parser.add_argument("--population", type=float, help="Total block population (e.g., 1100.0)")
    parser.add_argument("--households", type=float, help="Total block households (e.g., 420.0)")
    parser.add_argument("--median_income", type=float, help="Median income in tens of thousands USD (e.g., 8.3252)")
    parser.add_argument("--ocean_proximity", type=str, help="Ocean proximity category (<1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN)")

    args = parser.parse_args()

    if args.sample or (len(sys.argv) == 1):
        run_sample_predictions(args.model)
        return

    input_data = {}
    if args.json:
        if os.path.exists(args.json):
            with open(args.json, 'r') as f:
                input_data = json.load(f)
        else:
            input_data = json.loads(args.json)
    else:
        for f in FEATURE_NAMES:
            val = getattr(args, f)
            if val is not None:
                input_data[f] = val

    try:
        predicted_val = predict_single(input_data, model_path=args.model)
        print("\nInput Features Verified:")
        for k, v in input_data.items():
            print(f"  {k:25s}: {v}")
        print("\n" + "=" * 50)
        print(f"PREDICTED MEDIAN HOUSE VALUE: {format_usd(predicted_val)}")
        print("=" * 50)
    except Exception as e:
        print(f"\n[ERROR] Validation or Prediction Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

