import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility

# Number of records
n = 1000

# Existing features
patient_ids = range(1, n+1)
ages = np.random.randint(20, 91, n)
genders = np.random.choice(['Male', 'Female'], n)
tumor_sizes = np.where(np.random.rand(n) > 0.5, np.random.uniform(0, 100, n), 0)
biomarker_levels = np.random.uniform(0, 500, n)
genetic_mutations = np.random.choice(['None', 'BRCA1', 'TP53'], n, p=[0.7, 0.15, 0.15])
smoking_history = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])

# New medical report features
symptoms_options = ['None', 'Pain', 'Fatigue', 'Weight Loss', 'Cough', 'Pain, Fatigue', 'Weight Loss, Cough']
symptoms = np.random.choice(symptoms_options, n, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])

# Lab results: WBC (4-20), Hgb (8-18), Plt (100-500), Tumor_Marker (0-100)
wbc = np.random.uniform(4, 20, n)
hgb = np.random.uniform(8, 18, n)
plt = np.random.uniform(100, 500, n)
tumor_marker = np.random.uniform(0, 100, n)
lab_results = [f"{w:.1f}/{h:.1f}/{p:.0f}/{t:.1f}" for w, h, p, t in zip(wbc, hgb, plt, tumor_marker)]

imaging_findings = np.random.choice(['Normal', 'Mass Detected', 'Lymph Node Involvement'], n, p=[0.6, 0.3, 0.1])
family_history = np.random.choice(['Yes', 'No'], n, p=[0.2, 0.8])
comorbidities = np.random.choice(['None', 'Diabetes', 'Hypertension'], n, p=[0.7, 0.15, 0.15])
biopsy_result = np.random.choice(['Benign', 'Malignant', 'Inconclusive'], n, p=[0.5, 0.4, 0.1])

# Enhanced diagnosis logic: Incorporate multiple factors
diagnoses = []
for i in range(n):
    risk_score = 0
    risk_score += 2 if tumor_sizes[i] > 0 else 0
    risk_score += 1 if ages[i] > 50 else 0
    risk_score += 1 if smoking_history[i] == 'Yes' else 0
    risk_score += 1 if genetic_mutations[i] != 'None' else 0
    risk_score += 1 if symptoms[i] != 'None' else 0
    risk_score += 1 if imaging_findings[i] != 'Normal' else 0
    risk_score += 1 if family_history[i] == 'Yes' else 0
    risk_score += 1 if biopsy_result[i] == 'Malignant' else 0
    risk_score += 1 if tumor_marker[i] > 50 else 0  # High tumor marker
    diagnoses.append(1 if risk_score >= 4 else 0)  # Threshold for cancer

# Create DataFrame
df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Age': ages,
    'Gender': genders,
    'Tumor_Size_mm': tumor_sizes,
    'Biomarker_Level': biomarker_levels,
    'Genetic_Mutation': genetic_mutations,
    'Smoking_History': smoking_history,
    'Symptoms': symptoms,
    'Lab_Results': lab_results,
    'Imaging_Findings': imaging_findings,
    'Family_History': family_history,
    'Comorbidities': comorbidities,
    'Biopsy_Result': biopsy_result,
    'Diagnosis': diagnoses
})

# Save to CSV
df.to_csv('enhanced_synthetic_cancer_data.csv', index=False)
print("Generated 1000 enhanced synthetic patient records and saved to 'enhanced_synthetic_cancer_data.csv'")