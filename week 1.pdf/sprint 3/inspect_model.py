"""
Simple script to inspect AI_Makalu_OptimizedModel.pkl
Works by importing the pipeline code first
"""

import sys
import os

# Add the current directory to path so we can import the pipeline
sys.path.insert(0, 'd:/x pro/week 1.pdf/sprint 3')

# Import the custom classes from the pipeline
from AI_Makalu_Pipeline import FeatureEngineer, FeatureSelector

import joblib
import pandas as pd
import numpy as np

def inspect_model():
    """Load and inspect the model"""
    
    print("="*70)
    print("AI_MAKALU OPTIMIZED MODEL INSPECTION")
    print("="*70)
    
    # Load the model
    print("\n📦 Loading model...")
    model_path = 'd:/x pro/week 1.pdf/sprint 3/AI_Makalu_OptimizedModel.pkl'
    pipeline = joblib.load(model_path)
    print("✓ Model loaded successfully!")
    
    # Display pipeline structure
    print("\n" + "="*70)
    print("PIPELINE STRUCTURE")
    print("="*70)
    print(f"\nPipeline has {len(pipeline.steps)} steps:")
    for i, (name, transformer) in enumerate(pipeline.steps, 1):
        print(f"  Step {i}: {name} ({type(transformer).__name__})")
    
    # Get components
    feature_engineer = pipeline.named_steps['feature_engineer']
    feature_selector = pipeline.named_steps['feature_selector']
    scaler = pipeline.named_steps['scaler']
    classifier = pipeline.named_steps['classifier']
    
    # Feature Engineer details
    print("\n" + "="*70)
    print("STEP 1: FEATURE ENGINEER")
    print("="*70)
    print(f"Categorical features encoded: {len(feature_engineer.label_encoders)}")
    for col, encoder in feature_engineer.label_encoders.items():
        classes = list(encoder.classes_)
        print(f"  • {col}: {classes}")
    
    # Feature Selector details
    print("\n" + "="*70)
    print("STEP 2: FEATURE SELECTOR")
    print("="*70)
    print(f"Total features selected: {len(feature_selector.feature_columns)}\n")
    for i, feature in enumerate(feature_selector.feature_columns, 1):
        print(f"  {i:2d}. {feature}")
    
    # Scaler details
    print("\n" + "="*70)
    print("STEP 3: STANDARD SCALER")
    print("="*70)
    print(f"Features scaled: {scaler.n_features_in_}")
    print("\nFirst 5 features - Mean and Scale:")
    for i in range(min(5, len(feature_selector.feature_columns))):
        feature_name = feature_selector.feature_columns[i]
        print(f"  {feature_name:30s} Mean: {scaler.mean_[i]:8.2f}  Scale: {scaler.scale_[i]:8.2f}")
    
    # Classifier details
    print("\n" + "="*70)
    print("STEP 4: CLASSIFIER (LOGISTIC REGRESSION)")
    print("="*70)
    print(f"Model type: {type(classifier).__name__}")
    print(f"Features used: {classifier.n_features_in_}")
    print(f"Classes: {classifier.classes_} (0=Low Risk, 1=High Risk)")
    
    print("\nKey Hyperparameters:")
    print(f"  • C (regularization): {classifier.C}")
    print(f"  • Penalty: {classifier.penalty}")
    print(f"  • Solver: {classifier.solver}")
    print(f"  • Max iterations: {classifier.max_iter}")
    
    # Feature importance (coefficients)
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Model Coefficients)")
    print("="*70)
    
    coef = classifier.coef_[0]
    feature_importance = list(zip(feature_selector.feature_columns, coef))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop 10 Most Important Features:")
    print(f"{'Rank':<6} {'Feature':<35} {'Impact':<10} {'Coefficient':<12}")
    print("-" * 70)
    for i, (feature, coef_val) in enumerate(feature_importance[:10], 1):
        direction = "Increases" if coef_val > 0 else "Decreases"
        print(f"{i:<6} {feature:<35} {direction:<10} {coef_val:>11.4f}")
    
    # Example prediction
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION")
    print("="*70)
    
    # Create sample patient data
    sample_patient = pd.DataFrame({
        'Risk_Score': [75.0],
        'Gender': ['Male'],
        'Smoking': ['Yes'],
        'Alcohol_Consumption': ['High'],
        'Diabetes': ['Yes'],
        'Hypertension': ['Yes'],
        'Heart_Disease': ['No'],
        'Insurance_Type': ['Basic'],
        'Age': [65],
        'Height_cm': [175.0],
        'Weight_kg': [85.0],
        'BMI': [27.8],
        'Systolic_BP': [145],
        'Diastolic_BP': [90],
        'Heart_Rate': [75],
        'Temperature_F': [98.6],
        'Blood_Sugar': [150.0],
        'Cholesterol': [220.0],
        'Hemoglobin': [14.5],
        'Exercise_Hours_Week': [2.0],
        'Hospital_Visits_Year': [3]
    })
    
    print("\n👤 Sample Patient Profile:")
    print(f"  Age: {sample_patient['Age'].values[0]} years old")
    print(f"  Gender: {sample_patient['Gender'].values[0]}")
    print(f"  BMI: {sample_patient['BMI'].values[0]:.1f}")
    print(f"  Blood Pressure: {sample_patient['Systolic_BP'].values[0]}/{sample_patient['Diastolic_BP'].values[0]} mmHg")
    print(f"  Blood Sugar: {sample_patient['Blood_Sugar'].values[0]:.0f} mg/dL")
    print(f"  Cholesterol: {sample_patient['Cholesterol'].values[0]:.0f} mg/dL")
    print(f"  Smoking: {sample_patient['Smoking'].values[0]}")
    print(f"  Diabetes: {sample_patient['Diabetes'].values[0]}")
    print(f"  Hypertension: {sample_patient['Hypertension'].values[0]}")
    print(f"  Exercise: {sample_patient['Exercise_Hours_Week'].values[0]:.1f} hours/week")
    
    # Make prediction
    prediction = pipeline.predict(sample_patient)
    probability = pipeline.predict_proba(sample_patient)
    
    print(f"\n🎯 PREDICTION RESULT:")
    risk_level = "HIGH RISK ⚠️" if prediction[0] == 1 else "LOW RISK ✓"
    print(f"  Risk Level: {risk_level}")
    print(f"  Confidence: {max(probability[0]):.1%}")
    print(f"  Probabilities:")
    print(f"    - Low Risk:  {probability[0][0]:.1%}")
    print(f"    - High Risk: {probability[0][1]:.1%}")
    
    print("\n" + "="*70)
    print("INSPECTION COMPLETE ✓")
    print("="*70)
    print("\nThe model is ready for use!")
    print("To make predictions, use: pipeline.predict(patient_data)")

if __name__ == "__main__":
    inspect_model()
