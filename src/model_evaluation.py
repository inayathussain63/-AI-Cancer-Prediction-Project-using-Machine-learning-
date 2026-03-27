"""
Model evaluation module for the AI Cancer Prediction Project.

Computes classification metrics and generates visualisations for
trained models.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fitted model on the test set.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn classifier.
    X_test : array-like of shape (n_samples, n_features)
        Scaled test features.
    y_test : array-like of shape (n_samples,)
        True test labels.

    Returns
    -------
    dict
        Dictionary with keys 'accuracy', 'roc_auc', and 'report'.
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Malignant", "Benign"]
    )
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    return {"accuracy": accuracy, "roc_auc": roc_auc, "report": report}


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate multiple models and return their metrics.

    Parameters
    ----------
    models : dict
        Dictionary mapping model name to fitted model.
    X_test : array-like of shape (n_samples, n_features)
        Scaled test features.
    y_test : array-like of shape (n_samples,)
        True test labels.

    Returns
    -------
    dict
        Nested dictionary: {model_name: {metric_name: value}}.
    """
    return {name: evaluate_model(m, X_test, y_test) for name, m in models.items()}


def plot_confusion_matrix(model, X_test, y_test, model_name, output_dir="models"):
    """
    Save a confusion matrix heatmap for the given model.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn classifier.
    X_test : array-like of shape (n_samples, n_features)
        Scaled test features.
    y_test : array-like of shape (n_samples,)
        True test labels.
    model_name : str
        Name used in the plot title and file name.
    output_dir : str
        Directory where the PNG file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()

    safe_name = model_name.replace(" ", "_").lower()
    path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_roc_curves(models, X_test, y_test, output_dir="models"):
    """
    Save an ROC curve plot comparing all models.

    Parameters
    ----------
    models : dict
        Dictionary mapping model name to fitted model.
    X_test : array-like of shape (n_samples, n_features)
        Scaled test features.
    y_test : array-like of shape (n_samples,)
        True test labels.
    output_dir : str
        Directory where the PNG file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def print_summary(results):
    """
    Print a formatted summary of model evaluation results to stdout.

    Parameters
    ----------
    results : dict
        Nested dictionary returned by evaluate_all_models().
    """
    separator = "=" * 60
    for name, metrics in results.items():
        print(separator)
        print(f"Model: {name}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        if metrics["roc_auc"] is not None:
            print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"\n  Classification Report:\n{metrics['report']}")
    print(separator)
