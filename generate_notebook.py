"""
generate_notebook.py - Generate the model_analysis.ipynb notebook programmatically.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# ---- TITLE ----
cells.append(nbf.v4.new_markdown_cell("""# Inventory Insights - ML Model Analysis

This notebook contains:
1. **Exploratory Data Analysis (EDA)** - Sales distributions, seasonality, loss/waste patterns
2. **Feature Importance** - Which features drive the models
3. **Model Performance** - Actual vs Predicted, metrics comparison
4. **Inventory Decision Outputs** - Risk analysis and reorder recommendations
"""))

# ---- IMPORTS ----
cells.append(nbf.v4.new_code_cell("""import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.getcwd()) if 'notebooks' in os.getcwd() else os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

print("Libraries loaded successfully!")
"""))

# ---- LOAD DATA ----
cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading & Overview"))

cells.append(nbf.v4.new_code_cell("""from data_preprocessing import run_preprocessing
from feature_engineering import run_feature_engineering

merged, product_info, trans_df = run_preprocessing()
featured = run_feature_engineering(merged)

print(f"\\nDataset shape: {featured.shape}")
print(f"Products: {featured['product_id'].nunique()}")
print(f"Date range: {featured['date'].min().date()} to {featured['date'].max().date()}")
print(f"\\nProduct Info:")
display(product_info)
"""))

# ---- EDA: SALES DISTRIBUTION ----
cells.append(nbf.v4.new_markdown_cell("## 2. Exploratory Data Analysis (EDA)"))
cells.append(nbf.v4.new_markdown_cell("### 2.1 Daily Sales Distribution per Product"))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(3, 7, figsize=(24, 12))
axes = axes.flatten()

products = sorted(featured['product_id'].unique())
for i, pid in enumerate(products):
    ax = axes[i]
    data = featured[featured['product_id'] == pid]['daily_sales']
    data_nz = data[data > 0]
    ax.hist(data_nz, bins=30, color=sns.color_palette('husl', 21)[i], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_title(pid, fontsize=10, fontweight='bold')
    ax.set_xlabel('Daily Sales')
    ax.set_ylabel('Freq')

plt.suptitle('Daily Sales Distribution (Non-Zero Days)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ---- EDA: TIME SERIES ----
cells.append(nbf.v4.new_markdown_cell("### 2.2 Sales Time-Series (Top 6 Products)"))

cells.append(nbf.v4.new_code_cell("""# Top 6 products by total sales
top_products = featured.groupby('product_id')['daily_sales'].sum().nlargest(6).index

fig, axes = plt.subplots(3, 2, figsize=(18, 12))
axes = axes.flatten()

for i, pid in enumerate(top_products):
    ax = axes[i]
    data = featured[featured['product_id'] == pid][['date', 'daily_sales']].copy()
    data['rolling_7d'] = data['daily_sales'].rolling(7).mean()

    ax.plot(data['date'], data['daily_sales'], alpha=0.3, color='steelblue', linewidth=0.5)
    ax.plot(data['date'], data['rolling_7d'], color='darkblue', linewidth=1.5, label='7-day avg')
    ax.set_title(f'{pid}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend(fontsize=9)

plt.suptitle('Sales Time-Series with 7-Day Rolling Average', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ---- EDA: SEASONALITY ----
cells.append(nbf.v4.new_markdown_cell("### 2.3 Seasonality Patterns"))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Monthly
monthly = featured.groupby('month')['daily_sales'].mean()
axes[0].bar(monthly.index, monthly.values, color='coral', edgecolor='black', linewidth=0.5)
axes[0].set_title('Average Daily Sales by Month', fontweight='bold')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Avg Daily Sales')
axes[0].set_xticks(range(1, 13))

# Day of week
weekday = featured.groupby('weekday')['daily_sales'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1].bar(range(7), weekday.values, color='skyblue', edgecolor='black', linewidth=0.5)
axes[1].set_title('Average Daily Sales by Weekday', fontweight='bold')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(days)
axes[1].set_ylabel('Avg Daily Sales')

# Quarter
quarterly = featured.groupby('quarter')['daily_sales'].mean()
axes[2].bar(quarterly.index, quarterly.values, color='lightgreen', edgecolor='black', linewidth=0.5)
axes[2].set_title('Average Daily Sales by Quarter', fontweight='bold')
axes[2].set_xlabel('Quarter')
axes[2].set_ylabel('Avg Daily Sales')

plt.suptitle('Seasonality Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ---- LOAD MODELS ----
cells.append(nbf.v4.new_markdown_cell("## 3. Model Results & Feature Importance"))

cells.append(nbf.v4.new_code_cell("""from utils import load_model, DEMAND_MODEL_PATH, DECISION_MODEL_PATH

demand_bundle = load_model(DEMAND_MODEL_PATH)
decision_bundle = load_model(DECISION_MODEL_PATH)

print("Demand model methods:", demand_bundle.get('method_counts', {}))
print("Demand metrics:", demand_bundle.get('metrics', {}))
print("\\nDecision metrics:", decision_bundle.get('metrics', {}))
"""))

# ---- ACTUAL vs PREDICTED ----
cells.append(nbf.v4.new_markdown_cell("## 4. Model Performance Visualization"))
cells.append(nbf.v4.new_markdown_cell("### 4.1 Demand Predictions Summary"))

cells.append(nbf.v4.new_code_cell("""predictions = demand_bundle.get('predictions', {})
pred_df = pd.DataFrame([
    {'Product': k, 'Predicted_Demand_30d': v['predicted_demand_next_30_days'], 'Method': v['method']}
    for k, v in predictions.items()
]).sort_values('Predicted_Demand_30d', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

colors = {'SARIMAX': 'steelblue', 'RandomForest': 'coral', 'HistoricalAverage': 'gray'}
bar_colors = [colors.get(m, 'gray') for m in pred_df['Method']]

axes[0].barh(pred_df['Product'], pred_df['Predicted_Demand_30d'], color=bar_colors, edgecolor='black', linewidth=0.5)
axes[0].set_title('30-Day Demand Forecast by Product', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Predicted Units')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
axes[0].legend(handles=legend_elements, loc='lower right')

method_counts = pred_df['Method'].value_counts()
axes[1].pie(method_counts.values, labels=method_counts.index, autopct='%1.0f%%',
            colors=[colors.get(m, 'gray') for m in method_counts.index],
            startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Forecasting Methods Used', fontweight='bold', fontsize=13)

plt.suptitle('Demand Forecasting Results', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ---- RISK ANALYSIS ----
cells.append(nbf.v4.new_markdown_cell("### 4.2 Inventory Risk Analysis"))

cells.append(nbf.v4.new_code_cell("""decision_preds = decision_bundle.get('predictions', {})
dec_df = pd.DataFrame([
    {'Product': k, **v}
    for k, v in decision_preds.items()
]).sort_values('stock_duration_days')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

risk_counts = dec_df['stock_risk'].value_counts()
risk_colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
colors = [risk_colors.get(r, 'gray') for r in risk_counts.index]
axes[0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.0f%%',
            colors=colors, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Stock Risk Distribution', fontweight='bold', fontsize=13)

bar_colors = [risk_colors.get(r, 'gray') for r in dec_df['stock_risk']]
axes[1].barh(dec_df['Product'], dec_df['stock_duration_days'], color=bar_colors, edgecolor='black', linewidth=0.5)
axes[1].set_title('Stock Duration (Days)', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Days')
axes[1].axvline(x=30, color='red', linestyle='--', alpha=0.5, label='30-day threshold')
axes[1].legend()

axes[2].barh(dec_df['Product'], dec_df['reorder_quantity'], color=bar_colors, edgecolor='black', linewidth=0.5)
axes[2].set_title('Recommended Reorder Quantity', fontweight='bold', fontsize=13)
axes[2].set_xlabel('Units')

plt.suptitle('Inventory Decision Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

display(dec_df[['Product', 'current_stock', 'predicted_demand', 'stock_duration_days', 'reorder_quantity', 'stock_risk']])
"""))

nb.cells = cells

# Save notebook
notebook_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks')
os.makedirs(notebook_dir, exist_ok=True)
notebook_path = os.path.join(notebook_dir, 'model_analysis.ipynb')

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"[OK] Notebook created: {notebook_path}")
