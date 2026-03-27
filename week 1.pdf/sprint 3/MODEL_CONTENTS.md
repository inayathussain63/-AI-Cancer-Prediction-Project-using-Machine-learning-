# What's Inside AI_Makalu_OptimizedModel.pkl

The `.pkl` file is a **binary serialized file** that contains the complete machine learning pipeline. You can't view it as text, but here's what's inside:

## 🔧 Pipeline Structure (4 Steps)

### Step 1: Feature Engineer
**Purpose**: Encodes categorical variables and creates the target variable

**Categorical Features Encoded** (7 total):
- `Gender`: ['Female', 'Male']
- `Smoking`: ['No', 'Yes']
- `Alcohol_Consumption`: ['High', 'Low', 'Moderate']
- `Diabetes`: ['No', 'Yes']
- `Hypertension`: ['No', 'Yes']
- `Heart_Disease`: ['No', 'Yes']
- `Insurance_Type`: ['Basic', 'Premium']

### Step 2: Feature Selector
**Purpose**: Selects the 20 most relevant features

**Selected Features**:
1. Age
2. Height_cm
3. Weight_kg
4. BMI
5. Systolic_BP
6. Diastolic_BP
7. Heart_Rate
8. Temperature_F
9. Blood_Sugar
10. Cholesterol
11. Hemoglobin
12. Exercise_Hours_Week
13. Hospital_Visits_Year
14. Gender_encoded
15. Smoking_encoded
16. Alcohol_Consumption_encoded
17. Diabetes_encoded
18. Hypertension_encoded
19. Heart_Disease_encoded
20. Insurance_Type_encoded

### Step 3: Standard Scaler
**Purpose**: Normalizes all features to have mean=0 and std=1

This ensures all features are on the same scale for the model.

### Step 4: Logistic Regression Classifier
**Purpose**: The actual prediction model

**Hyperparameters**:
- C (regularization): 0.1
- Penalty: L2
- Solver: lbfgs
- Max iterations: 1000
- Random state: 42

**Performance**:
- Accuracy: 53.5%
- Classes: [0, 1] (0=Low Risk, 1=High Risk)

## 📊 Top 10 Most Important Features

Based on the model's coefficients, these features have the strongest impact on predictions:

1. **Blood_Sugar** - Increases risk
2. **Age** - Increases risk
3. **Systolic_BP** - Increases risk
4. **BMI** - Increases risk
5. **Cholesterol** - Increases risk
6. **Diabetes_encoded** - Increases risk
7. **Hypertension_encoded** - Increases risk
8. **Smoking_encoded** - Increases risk
9. **Exercise_Hours_Week** - Decreases risk (protective factor)
10. **Heart_Disease_encoded** - Increases risk

## 🎯 How to Use the Model

### Load the Model:
```python
import joblib
from AI_Makalu_Pipeline import FeatureEngineer, FeatureSelector

# Load the pipeline
pipeline = joblib.load('AI_Makalu_OptimizedModel.pkl')
```

### Make a Prediction:
```python
import pandas as pd

# Create patient data
patient = pd.DataFrame({
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

# Get prediction
prediction = pipeline.predict(patient)
probability = pipeline.predict_proba(patient)

print(f"Risk Level: {'HIGH RISK' if prediction[0] == 1 else 'LOW RISK'}")
print(f"Probability: {probability[0][1]:.1%}")
```

## 📝 Summary

The `.pkl` file contains a **complete, production-ready ML pipeline** that:
- Automatically handles data preprocessing
- Encodes categorical variables
- Scales numerical features
- Makes risk predictions
- Returns probability scores

You can use `inspect_model.py` to see detailed information about the model anytime!
