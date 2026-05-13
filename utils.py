"""
utils.py - Utility functions and configuration for the ML pipeline.

Contains:
- Path constants and configuration
- Model save/load functions using joblib
- Prediction generation for frontend integration
"""

import os
import joblib
import pandas as pd
import numpy as np

# ============================================================
# PATH CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)  # parent folder with CSVs
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# CSV file paths
STOCK_CSV = os.path.join(DATA_DIR, 'Stock Master.csv')
TRANSACTION_CSV = os.path.join(DATA_DIR, 'Transaction Master.csv')
LOSS_CSV = os.path.join(DATA_DIR, 'Lost Table.csv')
WASTE_CSV = os.path.join(DATA_DIR, 'Waste Table.csv')

# Model file paths
DEMAND_MODEL_PATH = os.path.join(MODELS_DIR, 'demand_model.joblib')
DECISION_MODEL_PATH = os.path.join(MODELS_DIR, 'decision_model.joblib')

# ============================================================
# DEFAULT PARAMETERS
# ============================================================
FORECAST_HORIZON = 30          # days to predict
TRAIN_TEST_SPLIT = 0.8         # 80/20 split
MIN_STOCK_MULTIPLIER = 0.15    # min_stock = median(Initial_Qty) * 0.15
SAFETY_STOCK_DAYS = 7          # extra buffer days for reorder
RANDOM_STATE = 42


# ============================================================
# MODEL SAVE / LOAD
# ============================================================
def save_model(model, filepath):
    """Save a model (or dict of models) to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[OK] Model saved to: {filepath}")


def load_model(filepath):
    """Load a model from disk using joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at: {filepath}")
    model = joblib.load(filepath)
    print(f"[OK] Model loaded from: {filepath}")
    return model


# ============================================================
# PREDICTION GENERATION (for backend integration)
# ============================================================
def generate_predictions(product_id=None):
    """
    Generate predictions for one or all products.
    Returns a list of dicts with keys matching frontend requirements.
    """
    demand_data = load_model(DEMAND_MODEL_PATH)
    decision_data = load_model(DECISION_MODEL_PATH)

    predictions = demand_data.get('predictions', {})
    decision_predictions = decision_data.get('predictions', {})

    results = []
    product_ids = [product_id] if product_id else list(predictions.keys())

    for pid in product_ids:
        demand_pred = predictions.get(pid, {})
        decision_pred = decision_predictions.get(pid, {})

        result = {
            'product_id': pid,
            'predicted_demand_next_30_days': int(demand_pred.get('predicted_demand_next_30_days', 0)),
            'stock_duration_days': round(decision_pred.get('stock_duration_days', 0), 1),
            'reorder_quantity': int(decision_pred.get('reorder_quantity', 0)),
            'reorder_time_days': round(decision_pred.get('reorder_time_days', 0), 1),
            'stock_risk': decision_pred.get('stock_risk', 'Unknown'),
        }
        results.append(result)

    return results


def print_predictions_table(results):
    """Pretty-print prediction results as a table."""
    if not results:
        print("No predictions available.")
        return
    df = pd.DataFrame(results)
    df = df.sort_values('stock_risk', ascending=True)
    print("\n" + "=" * 90)
    print("INVENTORY INSIGHTS - PREDICTION SUMMARY")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)


if __name__ == '__main__':
    try:
        results = generate_predictions()
        print_predictions_table(results)
    except FileNotFoundError as e:
        print(f"Models not yet trained: {e}")
        print("Run train_demand_model.py and train_decision_model.py first.")
