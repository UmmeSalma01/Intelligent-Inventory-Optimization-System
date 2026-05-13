"""
train_demand_model.py - Demand Forecasting Model Training.
"""

import pandas as pd
import numpy as np
import warnings
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_preprocessing import run_preprocessing
from feature_engineering import run_feature_engineering, get_demand_features
from evaluate_models import evaluate_regression
from utils import save_model, DEMAND_MODEL_PATH, FORECAST_HORIZON, TRAIN_TEST_SPLIT, RANDOM_STATE

warnings.filterwarnings('ignore')

def train_sarimax(product_series, forecast_days=30):
    """Attempt to fit SARIMAX on a product's daily sales time-series."""
    ts = product_series.copy()
    daily_nonzero = (ts > 0).sum()
    if daily_nonzero < 30:
        raise ValueError(f"Too few non-zero data points ({daily_nonzero})")
    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=200)
    forecast = results.forecast(steps=forecast_days)
    forecast = np.maximum(forecast, 0)
    forecast_sum = int(np.round(forecast.sum()))
    return forecast_sum, results


def train_random_forest(product_df, feature_cols, target_col='daily_sales', forecast_days=30):
    """Train a RandomForestRegressor on feature-engineered data."""
    df = product_df.sort_values('date').copy()
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    if len(train) < 30 or len(test) < 7:
        raise ValueError("Insufficient data for RandomForest training")
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=3, max_features='sqrt',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred = np.maximum(y_pred, 0)
    avg_daily_pred = np.mean(y_pred[-forecast_days:]) if len(y_pred) >= forecast_days else np.mean(y_pred)
    forecast_sum = int(np.round(avg_daily_pred * forecast_days))
    return forecast_sum, model, scaler, y_test, y_pred


def train_demand_model():
    """Train demand forecasting models for all products."""
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING DEMAND FORECASTING MODEL")
    print("=" * 60)

    merged, product_info, trans_df = run_preprocessing()
    featured = run_feature_engineering(merged)
    feature_cols = get_demand_features()
    products = featured['product_id'].unique()

    models = {}
    predictions = {}
    all_y_true = []
    all_y_pred = []
    method_counts = {'SARIMAX': 0, 'RandomForest': 0, 'Failed': 0}

    start_time = time.time()

    for i, pid in enumerate(products):
        product_df = featured[featured['product_id'] == pid].copy()
        product_df = product_df.sort_values('date')
        print(f"\n[{i+1}/{len(products)}] Training for {pid} ({len(product_df)} days)...", end=' ')
        try:
            ts = product_df.set_index('date')['daily_sales']
            ts = ts.asfreq('D', fill_value=0)
            forecast_sum, sarimax_model = train_sarimax(ts, FORECAST_HORIZON)
            models[pid] = {'type': 'SARIMAX', 'model': sarimax_model}
            predictions[pid] = {'predicted_demand_next_30_days': forecast_sum, 'method': 'SARIMAX'}
            method_counts['SARIMAX'] += 1
            print(f"SARIMAX [OK] (forecast: {forecast_sum} units)")
            in_sample = sarimax_model.fittedvalues
            split_idx = int(len(ts) * TRAIN_TEST_SPLIT)
            if split_idx < len(in_sample):
                all_y_true.extend(ts.values[split_idx:].tolist())
                all_y_pred.extend(np.maximum(in_sample.values[split_idx:], 0).tolist())
        except Exception as e:
            try:
                forecast_sum, rf_model, scaler, y_test, y_pred = train_random_forest(
                    product_df, feature_cols, 'daily_sales', FORECAST_HORIZON
                )
                models[pid] = {'type': 'RandomForest', 'model': rf_model, 'scaler': scaler, 'feature_cols': feature_cols}
                predictions[pid] = {'predicted_demand_next_30_days': forecast_sum, 'method': 'RandomForest'}
                method_counts['RandomForest'] += 1
                print(f"RandomForest [OK] (forecast: {forecast_sum} units)")
                all_y_true.extend(y_test.tolist())
                all_y_pred.extend(y_pred.tolist())
            except Exception as e2:
                avg_daily = product_df['daily_sales'].mean()
                forecast_sum = int(np.round(avg_daily * FORECAST_HORIZON))
                predictions[pid] = {'predicted_demand_next_30_days': max(forecast_sum, 0), 'method': 'HistoricalAverage'}
                method_counts['Failed'] += 1
                print(f"Fallback to avg (forecast: {forecast_sum} units) - {str(e2)[:60]}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DEMAND MODEL TRAINING COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  SARIMAX:        {method_counts['SARIMAX']} products")
    print(f"  RandomForest:   {method_counts['RandomForest']} products")
    print(f"  Fallback (avg): {method_counts['Failed']} products")

    if len(all_y_true) > 0:
        demand_metrics = evaluate_regression(all_y_true, all_y_pred, "Demand Forecasting (Aggregate)")
    else:
        demand_metrics = {}

    model_bundle = {
        'models': models, 'predictions': predictions,
        'metrics': demand_metrics, 'method_counts': method_counts,
    }
    save_model(model_bundle, DEMAND_MODEL_PATH)
    print(f"\n--- 30-Day Demand Forecast Summary ---")
    for pid in sorted(predictions.keys()):
        p = predictions[pid]
        print(f"  {pid:>10}: {p['predicted_demand_next_30_days']:>6} units  ({p['method']})")
    return model_bundle


if __name__ == '__main__':
    model_bundle = train_demand_model()
