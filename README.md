# AI Cancer Prediction Project using Machine Learning

A Machine Learning project that predicts the presence of cancer (malignant vs.
benign tumour) using clinical patient data from the
[Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
The project simulates a real-world AI development cycle used in healthcare.

---

## Project Structure

```
.
├── main.py                        # End-to-end ML pipeline script
├── requirements.txt               # Python dependencies
├── src/
│   ├── data_preprocessing.py      # Data loading, splitting, and scaling
│   ├── model_training.py          # Logistic Regression, Random Forest, SVM
│   ├── model_evaluation.py        # Metrics, confusion matrices, ROC curves
│   └── predict.py                 # Inference interface for new patient data
└── tests/
    └── test_cancer_prediction.py  # Unit tests for all modules
```

---

## Dataset

The **Wisconsin Breast Cancer Dataset** (built into scikit-learn) contains
569 samples and 30 numeric features computed from digitised images of a
fine needle aspirate (FNA) of a breast mass.

| Class | Label | Count |
|-------|-------|-------|
| Malignant | 0 | 212 |
| Benign | 1 | 357 |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
- Load and preprocess the dataset (80/20 stratified train/test split)
- Train three classifiers: Logistic Regression, Random Forest, and SVM
- Print accuracy, ROC-AUC, and classification reports for each model
- Save the best model and scaler to `models/`
- Generate confusion matrix PNGs and an ROC curve comparison plot in `models/`

### 3. Run unit tests

```bash
python -m pytest tests/ -v
```

---

## Models & Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~98.2 % | ~99.5 % |
| SVM (RBF kernel) | ~98.2 % | ~99.5 % |
| Random Forest | ~95.6 % | ~99.4 % |

---

## Predict on New Data

```python
from src.predict import predict

# 30 clinical feature values in Wisconsin dataset order
features = [...]  # replace with real patient data

result = predict(features, "models/best_model.pkl", "models/scaler.pkl")
print(result)
# {'prediction': 'Benign', 'probability_malignant': 0.0123, 'probability_benign': 0.9877}
```

---

## Development Cycle

1. **Data Ingestion** – `src/data_preprocessing.py`
2. **Preprocessing** – StandardScaler normalisation, stratified split
3. **Model Training** – `src/model_training.py`
4. **Evaluation** – `src/model_evaluation.py` (accuracy, ROC-AUC, confusion matrix)
5. **Persistence** – best model serialised with `joblib`
6. **Inference** – `src/predict.py`
7. **Testing** – `tests/test_cancer_prediction.py` (17 unit tests)
