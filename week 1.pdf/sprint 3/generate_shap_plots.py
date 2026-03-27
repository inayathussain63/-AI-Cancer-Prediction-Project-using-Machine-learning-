"""
Generate SHAP plots for the optimized model
"""

import sys
sys.path.insert(0, 'd:/x pro/week 1.pdf/sprint 3')

from AI_Makalu_Pipeline import FeatureEngineer, FeatureSelector, load_data
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("="*70)
print("GENERATING SHAP PLOTS")
print("="*70)

# Load data
print("\n📊 Loading data...")
data = load_data('d:/x pro/srint 2/final_cleaned_data.csv')
print(f"✓ Data loaded: {data.shape[0]} samples")

# Initialize transformers
feature_engineer = FeatureEngineer()
feature_selector = FeatureSelector()

# Apply feature engineering
data_engineered = feature_engineer.fit_transform(data)

# Select features
feature_selector.fit(data_engineered)
X = feature_selector.transform(data_engineered)
y = data_engineered['High_Risk']

feature_names = X.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

# Load the trained model
print("\n🤖 Loading trained model...")
pipeline = joblib.load('d:/x pro/week 1.pdf/sprint 3/AI_Makalu_OptimizedModel.pkl')
model = pipeline.named_steps['classifier']
print(f"✓ Model loaded: {type(model).__name__}")

# Create SHAP explainer
print("\n🔍 Creating SHAP explainer...")
# Use a sample for faster computation
sample_size = min(100, len(X_test_scaled))
X_sample = X_test_scaled.iloc[:sample_size]

explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_sample)

print(f"✓ SHAP values computed for {sample_size} samples")

# Generate SHAP summary plot
print("\n📈 Generating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot - Feature Impact on Risk Prediction', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('d:/x pro/week 1.pdf/sprint 3/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: shap_summary_plot.png")

# Generate SHAP feature importance plot
print("\n📊 Generating SHAP feature importance plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', show=False)
plt.title('SHAP Feature Importance - Mean Absolute Impact', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('d:/x pro/week 1.pdf/sprint 3/shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: shap_feature_importance.png")

# Calculate and display feature importance
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES (by SHAP values)")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(shap_values).mean(axis=0)
}).sort_values('Importance', ascending=False)

print(f"\n{'Rank':<6} {'Feature':<35} {'Mean |SHAP|':<12}")
print("-" * 70)
for i, row in enumerate(feature_importance.head(10).itertuples(), 1):
    print(f"{i:<6} {row.Feature:<35} {row.Importance:>11.4f}")

print("\n" + "="*70)
print("SHAP PLOTS GENERATED SUCCESSFULLY! ✓")
print("="*70)
print("\nGenerated files:")
print("  1. shap_summary_plot.png")
print("  2. shap_feature_importance.png")
print("\nThese plots show:")
print("  • Which features most impact predictions")
print("  • How feature values affect risk (high/low)")
print("  • Feature importance ranking")
print("="*70)
