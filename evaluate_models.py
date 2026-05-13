"""
evaluate_models.py - Model evaluation module.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


def evaluate_regression(y_true, y_pred, model_name="Model"):
    """Evaluate regression model performance."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')

    metrics = {
        'MAE': round(mae, 4), 'MSE': round(mse, 4), 'RMSE': round(rmse, 4),
        'MAPE': round(mape, 2), 'R2_Score': round(r2, 4),
    }

    print(f"\n{'=' * 50}")
    print(f"REGRESSION METRICS - {model_name}")
    print(f"{'=' * 50}")
    for key, val in metrics.items():
        unit = '%' if key == 'MAPE' else ''
        print(f"  {key:>10}: {val}{unit}")
    print(f"{'=' * 50}")
    return metrics


def evaluate_classification(y_true, y_pred, model_name="Model", labels=None):
    """Evaluate classification model performance."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    metrics = {
        'Accuracy': round(acc, 4), 'Precision_macro': round(prec, 4),
        'Recall_macro': round(rec, 4), 'F1_Score_macro': round(f1, 4),
    }

    print(f"\n{'=' * 50}")
    print(f"CLASSIFICATION METRICS - {model_name}")
    print(f"{'=' * 50}")
    for key, val in metrics.items():
        print(f"  {key:>18}: {val}")
    print(f"\n--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"--- Confusion Matrix ---")
    if labels:
        print(f"  Labels: {labels}")
    print(cm)
    print(f"{'=' * 50}")

    metrics['confusion_matrix'] = cm.tolist()
    return metrics


def evaluate_all_models(demand_results, decision_results):
    """Evaluate both models and print a combined summary."""
    print("\n" + "#" * 60)
    print("#  FULL MODEL EVALUATION REPORT")
    print("#" * 60)

    demand_metrics = evaluate_regression(
        demand_results['y_true'], demand_results['y_pred'],
        model_name="Demand Forecasting"
    )

    risk_metrics = evaluate_classification(
        decision_results['risk_true'], decision_results['risk_pred'],
        model_name="Stock Risk Classification", labels=['Low', 'Medium', 'High']
    )

    print("\n" + "#" * 60)
    print("#  EVALUATION COMPLETE")
    print("#" * 60)
    return {'demand_regression': demand_metrics, 'risk_classification': risk_metrics}


if __name__ == '__main__':
    print("Testing evaluation module...")
