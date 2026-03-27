import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add path to find the custom classes
sys.path.append('d:/x pro/week 1.pdf/sprint 3')
from AI_Makalu_ImprovedRF import AdvancedFeatureEngineer, AdvancedFeatureSelector

def verify():
    print("Verifying Improved Random Forest Performance...")
    
    # LOAD DATA
    data_path = 'd:/x pro/srint 2/final_cleaned_data.csv'
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        # Generate same dummy data as training
        from AI_Makalu_ImprovedRF import generate_dummy_data
        data = generate_dummy_data(1500)

    # PREPARE DATA
    # We need to replicate the split exactly or use a new test set
    # The pipeline handles engineering/selection, but we need y separate for evaluation
    
    # Load pipeline
    model_path = 'd:/x pro/week 1.pdf/sprint 3/AI_Makalu_ImprovedRF_Model.pkl'
    try:
        pipeline = joblib.load(model_path)
        print("✓ Loaded Improved RF Pipeline")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # To evaluate fairly, we should split first then predict, 
    # but the pipeline includes engineering/selection which might need the full dataframe structure
    # Actually, the pipeline expects raw dataframe or engineered one?
    # The pipeline saved in AI_Makalu_ImprovedRF.py is:
    # Pipeline([('feature_engineer', engineer), ('feature_selector', selector), ('scaler', scaler), ('classifier', best_rf)])
    
    # So we can pass raw 'data' columns (minus target) to predict.
    
    # Re-create Target for validation
    if 'Risk_Score' in data.columns:
        risk_threshold = data['Risk_Score'].median()
        y = (data['Risk_Score'] > risk_threshold).astype(int)
    else:
        print("No Risk_Score to evaluate against")
        return

    # Split (same seed)
    X = data.drop(columns=['High_Risk'] if 'High_Risk' in data.columns else [])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # EVALUATE IMPROVED MODEL
    print("\nEvaluated on Test Set (300 samples):")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    rf_acc = accuracy_score(y_test, y_pred)
    rf_auc = roc_auc_score(y_test, y_prob)
    rf_f1 = f1_score(y_test, y_pred)
    
    rf_model = pipeline.named_steps['classifier']
    with open('verification_results.txt', 'w') as f:
        f.write("IMPROVED RF Results:\n")
        f.write(f"  Accuracy: {rf_acc:.4f}\n")
        f.write(f"  AUC:      {rf_auc:.4f}\n")
        f.write(f"  F1 Score: {rf_f1:.4f}\n")
        f.write(f"\nBest Parameters:\n{rf_model.get_params()}\n")
    
    print("Results written to verification_results.txt")

if __name__ == "__main__":
    verify()
