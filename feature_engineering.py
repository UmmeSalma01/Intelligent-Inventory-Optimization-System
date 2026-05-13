"""
feature_engineering.py - Feature engineering for the ML pipeline.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def add_time_features(df):
    """Add calendar-based time features from the date column."""
    df = df.copy()
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    print("[OK] Time features added: day, month, weekday, is_weekend, day_of_year, week_of_year, quarter")
    return df


def add_lag_features(df, target_col='daily_sales', lags=[1, 7, 14]):
    """Add lag features for past sales values."""
    df = df.copy()
    df = df.sort_values(['product_id', 'date'])
    for lag in lags:
        col_name = f'lag_{lag}'
        df[col_name] = df.groupby('product_id')[target_col].shift(lag)
    lag_cols = [f'lag_{l}' for l in lags]
    df[lag_cols] = df[lag_cols].fillna(0)
    print(f"[OK] Lag features added: {lag_cols}")
    return df


def add_rolling_features(df, target_col='daily_sales', windows=[7, 14, 30]):
    """Add rolling mean and standard deviation features."""
    df = df.copy()
    df = df.sort_values(['product_id', 'date'])
    for window in windows:
        mean_col = f'rolling_mean_{window}d'
        std_col = f'rolling_std_{window}d'
        df[mean_col] = df.groupby('product_id')[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[std_col] = df.groupby('product_id')[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    rolling_cols = []
    for w in windows:
        rolling_cols.extend([f'rolling_mean_{w}d', f'rolling_std_{w}d'])
    df[rolling_cols] = df[rolling_cols].fillna(0)
    print(f"[OK] Rolling features added: {rolling_cols}")
    return df


def add_loss_waste_rolling(df, window=30):
    """Add rolling 30-day totals for loss and waste quantities."""
    df = df.copy()
    df = df.sort_values(['product_id', 'date'])
    df['total_loss_last_30_days'] = df.groupby('product_id')['daily_loss_qty'].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum()
    )
    df['total_waste_last_30_days'] = df.groupby('product_id')['daily_waste_qty'].transform(
        lambda x: x.rolling(window=window, min_periods=1).sum()
    )
    df['total_loss_last_30_days'] = df['total_loss_last_30_days'].fillna(0)
    df['total_waste_last_30_days'] = df['total_waste_last_30_days'].fillna(0)
    print(f"[OK] Loss/waste rolling features added (window={window}d)")
    return df


def add_inventory_features(df):
    """Add inventory-related features."""
    df = df.copy()
    avg_sales = df.groupby('product_id')['daily_sales'].mean().reset_index()
    avg_sales.columns = ['product_id', 'avg_daily_sales']
    df = df.merge(avg_sales, on='product_id', how='left')
    df['stock_to_sales_ratio'] = np.where(
        df['avg_daily_sales'] > 0,
        df['current_stock'] / df['avg_daily_sales'],
        999
    )
    print("[OK] Inventory features added: avg_daily_sales, stock_to_sales_ratio")
    return df


def run_feature_engineering(merged_df):
    """Execute the full feature engineering pipeline."""
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    df = merged_df.copy()
    df = add_time_features(df)
    df = add_lag_features(df, target_col='daily_sales', lags=[1, 7, 14])
    df = add_rolling_features(df, target_col='daily_sales', windows=[7, 14, 30])
    df = add_loss_waste_rolling(df, window=30)
    df = add_inventory_features(df)

    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"[!!] Dropped {dropped} rows with remaining NaN values")

    print(f"\n[OK] Feature engineering complete: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_demand_features():
    """Return the list of feature columns for the demand model."""
    return [
        'day', 'month', 'weekday', 'is_weekend', 'day_of_year', 'quarter',
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7d', 'rolling_mean_14d', 'rolling_mean_30d',
        'rolling_std_7d', 'rolling_std_14d', 'rolling_std_30d',
        'total_loss_last_30_days', 'total_waste_last_30_days',
    ]


def get_decision_features():
    """Return the list of feature columns for the decision model."""
    return [
        'current_stock', 'min_stock_level', 'avg_daily_sales',
        'stock_to_sales_ratio', 'lead_time',
        'total_loss_last_30_days', 'total_waste_last_30_days',
        'rolling_mean_7d', 'rolling_mean_30d',
        'rolling_std_7d',
        'predicted_demand',
    ]


if __name__ == '__main__':
    from data_preprocessing import run_preprocessing
    merged, product_info, _ = run_preprocessing()
    featured = run_feature_engineering(merged)
    print("\nSample featured data (first 3 rows):")
    print(featured.head(3).to_string())
