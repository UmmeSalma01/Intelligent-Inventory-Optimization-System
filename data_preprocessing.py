"""
data_preprocessing.py - Data loading, cleaning, and merging for the ML pipeline.
"""

import pandas as pd
import numpy as np
import warnings
from utils import STOCK_CSV, TRANSACTION_CSV, LOSS_CSV, WASTE_CSV, MIN_STOCK_MULTIPLIER

warnings.filterwarnings('ignore')


def load_stock_data():
    """Load and clean the Stock Master CSV."""
    df = pd.read_csv(STOCK_CSV)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns={
        'Stock Code': 'product_id', 'Product Code': 'batch_id',
        'Manufacturing Date': 'mfg_date', 'Expiry Date': 'expiry_date',
        'Initial_Qty': 'initial_qty', 'Remaining_Qty': 'remaining_qty',
        'Arrival Date': 'arrival_date', 'Shelf Life': 'shelf_life',
        'Lead Time': 'lead_time',
    })
    for col in ['mfg_date', 'expiry_date', 'arrival_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['lead_time'] = df.groupby('product_id')['lead_time'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df['shelf_life'] = df.groupby('product_id')['shelf_life'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    print(f"[OK] Stock data loaded: {df.shape[0]} rows, {df['product_id'].nunique()} products")
    return df


def load_transaction_data():
    """Load and clean the Transaction Master CSV."""
    df = pd.read_csv(TRANSACTION_CSV)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns={
        'Stock Code': 'product_id', 'Units Sold': 'units_sold',
        'Date': 'date', 'Category': 'category', 'Sub Category': 'sub_category',
        'Sales': 'sales', 'Discount': 'discount', 'Profit': 'profit',
        'Day of week': 'day_of_week', 'IsWeekend': 'is_weekend',
        'Weather': 'weather', 'Unit Cost': 'unit_cost',
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'units_sold'])
    df['units_sold'] = df['units_sold'].astype(int)
    print(f"[OK] Transaction data loaded: {df.shape[0]} rows, "
          f"date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def load_loss_data():
    """Load and clean the Lost Table CSV."""
    df = pd.read_csv(LOSS_CSV)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns={
        'Date': 'date', 'Stock Code': 'product_id',
        'Lost_Qty': 'lost_qty', 'Lost_Profit': 'lost_profit',
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    print(f"[OK] Loss data loaded: {df.shape[0]} rows")
    return df


def load_waste_data():
    """Load and clean the Waste Table CSV."""
    df = pd.read_csv(WASTE_CSV)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns={
        'Date': 'date', 'Stock Code': 'product_id',
        'Expired_Qty': 'expired_qty', 'Waste_Lost_Profit': 'waste_profit',
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    print(f"[OK] Waste data loaded: {df.shape[0]} rows")
    return df


def aggregate_daily_sales(trans_df):
    """Aggregate transaction data to daily total units sold per product."""
    daily = trans_df.groupby(['product_id', 'date']).agg(
        daily_sales=('units_sold', 'sum'),
        daily_revenue=('sales', 'sum'),
        daily_transactions=('units_sold', 'count'),
    ).reset_index()

    date_min = daily['date'].min()
    date_max = daily['date'].max()
    all_dates = pd.date_range(start=date_min, end=date_max, freq='D')
    products = daily['product_id'].unique()

    idx = pd.MultiIndex.from_product([products, all_dates], names=['product_id', 'date'])
    daily_complete = daily.set_index(['product_id', 'date']).reindex(idx, fill_value=0).reset_index()

    print(f"[OK] Daily sales aggregated: {daily_complete.shape[0]} rows "
          f"({len(products)} products x {len(all_dates)} days)")
    return daily_complete


def compute_product_info(stock_df):
    """Compute per-product static features."""
    product_info = []
    for pid, group in stock_df.groupby('product_id'):
        group = group.sort_values('arrival_date')
        latest = group.iloc[-1]
        current_stock = group['remaining_qty'].sum()
        info = {
            'product_id': pid,
            'current_stock': int(current_stock),
            'min_stock_level': int(np.ceil(group['initial_qty'].median() * MIN_STOCK_MULTIPLIER)),
            'lead_time': float(latest['lead_time']),
            'shelf_life': float(latest['shelf_life']),
            'avg_initial_qty': float(group['initial_qty'].mean()),
            'total_batches': len(group),
        }
        product_info.append(info)
    product_df = pd.DataFrame(product_info)
    print(f"[OK] Product info computed for {len(product_df)} products")
    return product_df


def aggregate_daily_loss(loss_df):
    """Aggregate loss data to daily totals per product."""
    return loss_df.groupby(['product_id', 'date']).agg(
        daily_loss_qty=('lost_qty', 'sum')).reset_index()


def aggregate_daily_waste(waste_df):
    """Aggregate waste data to daily totals per product."""
    return waste_df.groupby(['product_id', 'date']).agg(
        daily_waste_qty=('expired_qty', 'sum')).reset_index()


def merge_all_data(daily_sales, product_info, daily_loss, daily_waste):
    """Merge daily sales with product info, loss, and waste data."""
    merged = daily_sales.merge(product_info, on='product_id', how='left')
    merged = merged.merge(daily_loss, on=['product_id', 'date'], how='left')
    merged['daily_loss_qty'] = merged['daily_loss_qty'].fillna(0)
    merged = merged.merge(daily_waste, on=['product_id', 'date'], how='left')
    merged['daily_waste_qty'] = merged['daily_waste_qty'].fillna(0)
    merged = merged.sort_values(['product_id', 'date']).reset_index(drop=True)
    print(f"[OK] All data merged: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


def run_preprocessing():
    """Execute the full preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    stock_df = load_stock_data()
    trans_df = load_transaction_data()
    loss_df = load_loss_data()
    waste_df = load_waste_data()

    daily_sales = aggregate_daily_sales(trans_df)
    product_info = compute_product_info(stock_df)
    daily_loss = aggregate_daily_loss(loss_df)
    daily_waste = aggregate_daily_waste(waste_df)
    merged = merge_all_data(daily_sales, product_info, daily_loss, daily_waste)

    print(f"\n--- Data Quality Report ---")
    print(f"Total rows: {merged.shape[0]}")
    missing = merged.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"Missing values:\n{missing}")
    else:
        print("Missing values: None")
    print(f"Products: {merged['product_id'].nunique()}")
    print(f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")

    return merged, product_info, trans_df


if __name__ == '__main__':
    merged, product_info, _ = run_preprocessing()
    print("\nSample data (first 5 rows):")
    print(merged.head().to_string())
