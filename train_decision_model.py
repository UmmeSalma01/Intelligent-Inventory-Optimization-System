"""
train_decision_model.py - Inventory Decision Model Training.
"""

import pandas as pd
import numpy as np
import warnings
import time
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from data_preprocessing import run_preprocessing
from feature_engineering import run_feature_engineering, get_decision_features
from evaluate_models import evaluate_classification
from utils import (
    save_model, load_model, DEMAND_MODEL_PATH, DECISION_MODEL_PATH,
    FORECAST_HORIZON, SAFETY_STOCK_DAYS, RANDOM_STATE
)

warnings.filterwarnings('ignore')


def create_risk_labels(df):
    """Engineer stock risk labels based on lead time and stock duration."""
    df = df.copy()
    df['stock_duration_days'] = np.where(
        df['avg_daily_sales'] > 0,
        df['current_stock'] / df['avg_daily_sales'],
        999
    )
    conditions = [
        df['stock_duration_days'] < df['lead_time'],
        df['stock_duration_days'] < df['lead_time'] * 2,
        df['stock_duration_days'] >= df['lead_time'] * 2,
    ]
    choices = ['High', 'Medium', 'Low']
    df['stock_risk'] = np.select(conditions, choices, default='Low')
    risk_dist = df.groupby('product_id')['stock_risk'].last().value_counts()
    print(f"[OK] Risk labels created. Distribution (per product, latest):")
    for risk, count in risk_dist.items():
        print(f"    {risk}: {count}")
    return df


def compute_decision_outputs(product_info, demand_predictions):
    """Compute deterministic decision outputs per product."""
    results = {}
    for _, row in product_info.iterrows():
        pid = row['product_id']
        current_stock = row['current_stock']
        min_stock_level = row['min_stock_level']
        lead_time = row['lead_time']
        avg_daily_sales = row.get('avg_daily_sales', 0)
        pred = demand_predictions.get(pid, {})
        predicted_demand = pred.get('predicted_demand_next_30_days', 0)
        daily_demand_rate = predicted_demand / FORECAST_HORIZON if FORECAST_HORIZON > 0 else 0
        effective_daily_rate = max(daily_demand_rate, avg_daily_sales) if avg_daily_sales > 0 else daily_demand_rate

        stock_duration_days = current_stock / effective_daily_rate if effective_daily_rate > 0 else 999.0
        demand_during_lead = effective_daily_rate * (lead_time + SAFETY_STOCK_DAYS)
        reorder_quantity = max(0, int(np.ceil(demand_during_lead + min_stock_level - current_stock)))
        reorder_point = effective_daily_rate * lead_time + min_stock_level
        reorder_time_days = (current_stock - reorder_point) / effective_daily_rate if (effective_daily_rate > 0 and current_stock > reorder_point) else 0

        if stock_duration_days < lead_time:
            stock_risk = 'High'
        elif stock_duration_days < lead_time * 2:
            stock_risk = 'Medium'
        else:
            stock_risk = 'Low'

        results[pid] = {
            'stock_duration_days': round(stock_duration_days, 1),
            'reorder_quantity': reorder_quantity,
            'reorder_time_days': round(max(reorder_time_days, 0), 1),
            'stock_risk': stock_risk,
            'current_stock': current_stock,
            'predicted_demand': predicted_demand,
            'daily_demand_rate': round(effective_daily_rate, 2),
        }
    return results


def train_risk_classifier(featured_df):
    """Train an XGBoost classifier for stock risk prediction."""
    print("\n--- Training XGBoost Risk Classifier ---")
    df = create_risk_labels(featured_df)
    feature_cols = [
        'current_stock', 'min_stock_level', 'avg_daily_sales',
        'stock_to_sales_ratio', 'lead_time',
        'total_loss_last_30_days', 'total_waste_last_30_days',
        'rolling_mean_7d', 'rolling_mean_30d', 'rolling_std_7d',
    ]
    train_rows = []
    for pid, group in df.groupby('product_id'):
        group = group.sort_values('date')
        periodic = group.iloc[::30]
        recent = group.tail(90)
        combined = pd.concat([periodic, recent]).drop_duplicates()
        train_rows.append(combined)

    train_df = pd.concat(train_rows, ignore_index=True)
    le = LabelEncoder()
    le.classes_ = np.array(['Low', 'Medium', 'High'])
    train_df['risk_encoded'] = le.transform(train_df['stock_risk'])

    X = train_df[feature_cols].values
    y = train_df['risk_encoded'].values
    
    # XGBoost requires classes to be 0 to n_classes-1.
    # Since our dataset might only have e.g. "Low" and "High", 'y' might just have [0, 2].
    # We must remap them to [0, 1] for XGBoost to be happy.
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    # Create mapping from actual label encoding to 0, 1... n
    label_map = {old: new for new, old in enumerate(unique_classes)}
    y_mapped = np.array([label_map[val] for val in y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_weights = compute_sample_weight('balanced', y_mapped)

    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_mapped[:split_idx], y_mapped[split_idx:]
    w_train = sample_weights[:split_idx]

    param_grid = {
        'max_depth': [3, 5, 7], 'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1], 'min_child_weight': [1, 3],
    }
    xgb_base = XGBClassifier(
        objective='multi:softmax' if num_classes > 2 else 'binary:logistic',
        num_class=num_classes if num_classes > 2 else None,
        random_state=RANDOM_STATE,
        use_label_encoder=False, 
        eval_metric='mlogloss' if num_classes > 2 else 'logloss', 
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # If using string scoring 'f1_macro' might complain about multiclass vs binary,
    # but 'f1_macro' usually works for both.
    grid_search = GridSearchCV(xgb_base, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train, sample_weight=w_train)

    best_model = grid_search.best_estimator_
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1 (macro): {grid_search.best_score_:.4f}")

    y_pred = best_model.predict(X_test)
    
    # Map back to original LabelEncoder values
    inverse_map = {new: old for old, new in label_map.items()}
    y_test_original = np.array([inverse_map[val] for val in y_test])
    y_pred_original = np.array([inverse_map[val] for val in y_pred])
    
    y_test_labels = le.inverse_transform(y_test_original)
    y_pred_labels = le.inverse_transform(y_pred_original)

    clf_metrics = evaluate_classification(
        y_test_labels, y_pred_labels,
        model_name="Stock Risk (XGBoost)", labels=['Low', 'Medium', 'High']
    )
    return best_model, scaler, le, feature_cols, clf_metrics, y_test_labels, y_pred_labels


def train_decision_model():
    """Train the inventory decision model."""
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING INVENTORY DECISION MODEL")
    print("=" * 60)

    start_time = time.time()
    print("\n[1/4] Loading demand model predictions...")
    demand_bundle = load_model(DEMAND_MODEL_PATH)
    demand_predictions = demand_bundle['predictions']

    print("\n[2/4] Preprocessing & feature engineering...")
    merged, product_info, _ = run_preprocessing()
    featured = run_feature_engineering(merged)

    avg_sales = featured.groupby('product_id')['daily_sales'].mean().reset_index()
    avg_sales.columns = ['product_id', 'avg_daily_sales']
    product_info = product_info.merge(avg_sales, on='product_id', how='left')
    product_info['avg_daily_sales'] = product_info['avg_daily_sales'].fillna(0)

    print("\n[3/4] Computing decision outputs...")
    decision_predictions = compute_decision_outputs(product_info, demand_predictions)

    print("\n[4/4] Training risk classifier...")
    xgb_model, scaler, label_encoder, feature_cols, clf_metrics, y_test, y_pred = \
        train_risk_classifier(featured)

    elapsed = time.time() - start_time
    model_bundle = {
        'xgb_model': xgb_model, 'scaler': scaler, 'label_encoder': label_encoder,
        'feature_cols': feature_cols, 'predictions': decision_predictions,
        'metrics': clf_metrics, 'product_info': product_info.to_dict('records'),
    }
    save_model(model_bundle, DECISION_MODEL_PATH)

    print(f"\n{'=' * 60}")
    print(f"DECISION MODEL TRAINING COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 60}")

    print(f"\n--- Inventory Decision Summary ---")
    print(f"{'Product':<12} {'Stock':>6} {'Demand':>7} {'Duration':>9} {'Reorder':>8} {'ReorderIn':>9} {'Risk':>6}")
    print("-" * 70)
    for pid in sorted(decision_predictions.keys()):
        d = decision_predictions[pid]
        print(f"{pid:<12} {d['current_stock']:>6} {d['predicted_demand']:>7} "
              f"{d['stock_duration_days']:>8.1f}d {d['reorder_quantity']:>8} "
              f"{d['reorder_time_days']:>8.1f}d {d['stock_risk']:>6}")

    return model_bundle


if __name__ == '__main__':
    model_bundle = train_decision_model()
