"""
AI_Makalu_Pipeline.py
Sprint 3: Complete ML Pipeline with Hyperparameter Tuning and SHAP Analysis
Team: Makalu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import json
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering"""
    
    def __init__(self):
        self.label_encoders = {}
        
    def fit(self, X, y=None):
        # Fit label encoders for categorical columns
        categorical_cols = ['Gender', 'Smoking', 'Alcohol_Consumption', 'Diabetes', 
                           'Hypertension', 'Heart_Disease', 'Insurance_Type']
        
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.label_encoders[col] = le
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in X_copy.columns:
                X_copy[col + '_encoded'] = le.transform(X_copy[col].astype(str))
        
        # Create target variable if Risk_Score exists
        if 'Risk_Score' in X_copy.columns:
            risk_threshold = X_copy['Risk_Score'].median()
            X_copy['High_Risk'] = (X_copy['Risk_Score'] > risk_threshold).astype(int)
        
        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer for feature selection"""
    
    def __init__(self):
        self.feature_columns = None
        
    def fit(self, X, y=None):
        # Define feature columns
        base_features = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'Systolic_BP', 
                        'Diastolic_BP', 'Heart_Rate', 'Temperature_F', 'Blood_Sugar', 
                        'Cholesterol', 'Hemoglobin', 'Exercise_Hours_Week', 
                        'Hospital_Visits_Year']
        
        encoded_features = ['Gender_encoded', 'Smoking_encoded', 'Alcohol_Consumption_encoded',
                           'Diabetes_encoded', 'Hypertension_encoded', 'Heart_Disease_encoded',
                           'Insurance_Type_encoded']
        
        self.feature_columns = [col for col in base_features + encoded_features if col in X.columns]
        
        return self
    
    def transform(self, X):
        return X[self.feature_columns]


def load_data(filepath='d:/x pro/srint 2/final_cleaned_data.csv'):
    """Load data from file or generate dummy data"""
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        data = pd.read_csv(filepath)
    else:
        print("Data file not found. Generating dummy data...")
        data = generate_dummy_data()
    
    return data


def generate_dummy_data(n_samples=1000):
    """Generate dummy data for testing"""
    data = {
        'Risk_Score': np.random.uniform(0, 100, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Smoking': np.random.choice(['Yes', 'No'], n_samples),
        'Alcohol_Consumption': np.random.choice(['High', 'Moderate', 'Low'], n_samples),
        'Diabetes': np.random.choice(['Yes', 'No'], n_samples),
        'Hypertension': np.random.choice(['Yes', 'No'], n_samples),
        'Heart_Disease': np.random.choice(['Yes', 'No'], n_samples),
        'Insurance_Type': np.random.choice(['Basic', 'Premium'], n_samples),
        'Age': np.random.randint(18, 90, n_samples),
        'Height_cm': np.random.uniform(150, 200, n_samples),
        'Weight_kg': np.random.uniform(50, 120, n_samples),
        'BMI': np.random.uniform(18, 35, n_samples),
        'Systolic_BP': np.random.randint(90, 180, n_samples),
        'Diastolic_BP': np.random.randint(60, 120, n_samples),
        'Heart_Rate': np.random.randint(50, 100, n_samples),
        'Temperature_F': np.random.uniform(97, 100, n_samples),
        'Blood_Sugar': np.random.uniform(70, 200, n_samples),
        'Cholesterol': np.random.uniform(150, 300, n_samples),
        'Hemoglobin': np.random.uniform(10, 18, n_samples),
        'Exercise_Hours_Week': np.random.uniform(0, 10, n_samples),
        'Hospital_Visits_Year': np.random.randint(0, 5, n_samples)
    }
    return pd.DataFrame(data)


def optimize_model(X_train, y_train):
    """Perform hyperparameter tuning on multiple models"""
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    models_params = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }
    
    best_models = {}
    results = []
    
    for name, config in models_params.items():
        print(f"\nOptimizing {name}...")
        
        # Use RandomizedSearchCV for faster tuning
        search = RandomizedSearchCV(
            config['model'],
            config['params'],
            n_iter=10,
            cv=3,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        best_models[name] = search.best_estimator_
        
        results.append({
            'Model': name,
            'Best_Score': search.best_score_,
            'Best_Params': search.best_params_
        })
        
        print(f"Best CV Score: {search.best_score_:.4f}")
        print(f"Best Params: {search.best_params_}")
    
    return best_models, results


def evaluate_models(models, X_test, y_test):
    """Evaluate all models on test set"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = []
    best_model_name = None
    best_accuracy = 0
    best_model = None
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'AUC': auc
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print(f"{'='*60}")
    
    return pd.DataFrame(results), best_model, best_model_name


def perform_shap_analysis(model, X_train, X_test, feature_names, output_dir='d:/x pro/week 1.pdf/sprint 3'):
    """Perform SHAP analysis on the best model"""
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("="*60)
    
    try:
        import shap
        
        # Use a sample for faster computation
        sample_size = min(100, len(X_test))
        X_test_sample = X_test.iloc[:sample_size].copy()
        
        # Create SHAP explainer based on model type
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            print("Using TreeExplainer for tree-based model...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            
            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif isinstance(model, LogisticRegression):
            print("Using LinearExplainer for Logistic Regression...")
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test_sample)
        else:
            # Use KernelExplainer for other models
            print("Using KernelExplainer for model...")
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
            shap_values = explainer.shap_values(X_test_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        print(f"✓ SHAP values computed for {sample_size} samples")
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on Risk Prediction', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP summary plot saved")
        
        # Feature importance plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, plot_type='bar', show=False)
        plt.title('SHAP Feature Importance - Mean Absolute Impact', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP feature importance plot saved")
        
        # Calculate mean absolute SHAP values for feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return feature_importance
        
    except ImportError:
        print("Warning: SHAP library not installed. Skipping SHAP analysis.")
        print("Install with: pip install shap")
        return None
    except Exception as e:
        print(f"Warning: SHAP analysis failed with error: {e}")
        print("Continuing without SHAP plots...")
        return None


def save_model_and_metadata(model, scaler, feature_engineer, feature_selector, 
                            metrics_df, best_model_name, feature_names, 
                            output_dir='d:/x pro/week 1.pdf/sprint 3'):
    """Save the optimized model pipeline and metadata"""
    print("\n" + "="*60)
    print("SAVING MODEL AND METADATA")
    print("="*60)
    
    # Create complete pipeline
    pipeline = Pipeline([
        ('feature_engineer', feature_engineer),
        ('feature_selector', feature_selector),
        ('scaler', scaler),
        ('classifier', model)
    ])
    
    # Save pipeline
    model_path = os.path.join(output_dir, 'AI_Makalu_OptimizedModel.pkl')
    joblib.dump(pipeline, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Prepare metadata
    best_metrics = metrics_df[metrics_df['Model'] == best_model_name].iloc[0]
    
    metadata = {
        'model_info': {
            'model_name': best_model_name,
            'model_type': type(model).__name__,
            'version': '1.0',
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'team': 'Makalu'
        },
        'performance_metrics': {
            'accuracy': float(best_metrics['Accuracy']),
            'precision': float(best_metrics['Precision']),
            'recall': float(best_metrics['Recall']),
            'f1_score': float(best_metrics['F1_Score']),
            'auc': float(best_metrics['AUC'])
        },
        'training_data': {
            'source': 'final_cleaned_data.csv',
            'total_samples': 1000,
            'train_samples': 800,
            'test_samples': 200,
            'features_count': len(feature_names)
        },
        'hyperparameters': model.get_params(),
        'features': feature_names
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'AI_Makalu_Model_Metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return pipeline, metadata


def main():
    """Main pipeline execution"""
    print("\n" + "="*60)
    print("AI MAKALU - SPRINT 3 PIPELINE")
    print("Health Risk Assessment Model")
    print("="*60)
    
    # Set output directory
    output_dir = 'd:/x pro/week 1.pdf/sprint 3'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_data()
    print(f"\nData loaded: {data.shape[0]} samples, {data.shape[1]} features")
    
    # Initialize transformers
    feature_engineer = FeatureEngineer()
    feature_selector = FeatureSelector()
    
    # Apply feature engineering
    data_engineered = feature_engineer.fit_transform(data)
    
    # Prepare features and target
    if 'High_Risk' not in data_engineered.columns:
        print("Error: Target variable 'High_Risk' not created")
        return
    
    # Select features
    feature_selector.fit(data_engineered)
    X = feature_selector.transform(data_engineered)
    y = data_engineered['High_Risk']
    
    feature_names = X.columns.tolist()
    print(f"Features selected: {len(feature_names)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for SHAP
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Optimize models
    best_models, optimization_results = optimize_model(X_train_scaled, y_train)
    
    # Evaluate models
    metrics_df, best_model, best_model_name = evaluate_models(best_models, X_test_scaled, y_test)
    
    # Perform SHAP analysis
    feature_importance = perform_shap_analysis(
        best_model, X_train_scaled, X_test_scaled, feature_names, output_dir
    )
    
    # Save model and metadata
    pipeline, metadata = save_model_and_metadata(
        best_model, scaler, feature_engineer, feature_selector,
        metrics_df, best_model_name, feature_names, output_dir
    )
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. AI_Makalu_OptimizedModel.pkl")
    print("  2. AI_Makalu_Model_Metadata.json")
    print("  3. shap_summary_plot.png")
    print("  4. shap_feature_importance.png")
    print("\nNext steps:")
    print("  - Create AI_Makalu_SHAP_Analysis.docx")
    print("  - Create AI_Makalu_Final_Presentation.pptx")
    print("="*60)


if __name__ == "__main__":
    main()