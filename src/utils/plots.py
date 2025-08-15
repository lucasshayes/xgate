import os
import json
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, PrecisionRecallDisplay, roc_auc_score, RocCurveDisplay, roc_curve

def cm_plot(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    path: str = "reports/confusion_matrix.png",
) -> None:
    """
    Plot a confusion matrix to desired location.

    Args:
        cm (np.ndarray): Confusion matrix.
        labels (list[str]): List of class labels.
        title (str, optional): Title of the plot. Defaults to "Confusion Matrix".
        path (str, optional): Path to save the plot. Defaults to "reports/confusion_matrix.png".
    
    Raises:
        ValueError: If the input arrays or labels are invalid.
    """
    if not isinstance(cm, np.ndarray) or cm.ndim != 2:
        raise ValueError("Invalid input: cm must be a 2D numpy array.")
    
    if not labels or not isinstance(labels, list):
        raise ValueError("Invalid labels provided.")

    plt.figure(figsize=(8, 6))
    cmap = sns.color_palette("flare", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(path)

def precision_recall_plot(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], path: str = "reports/precision_recall.png") -> None:
    """Plot precision-recall curves for each class.

    Args:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred (np.ndarray): Predicted probabilities for each class.
        labels (list[str]): List of class labels.
        path (str, optional): Path to save the plot. Defaults to "reports/precision_recall.png".
    
    Raises:
        ValueError: If the input arrays or labels are invalid.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("Invalid input: y_true and y_pred must be numpy arrays.")

    if not y_true.ndim == 2 or not y_pred.ndim == 2:
        raise ValueError("Invalid input: y_true and y_pred must be 2D numpy arrays.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Incompatible input shapes: y_true and y_pred must have the same number of samples.")

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("Incompatible input shapes: y_true and y_pred must have the same number of classes.")

    if not labels or not isinstance(labels, list):
        raise ValueError("Invalid labels provided.")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("flare", len(labels))

    precision, recall, average_precision = {}, {}, {}
    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])
        
        ax.plot(recall[i], precision[i], label=f"{labels[i]} (AP={average_precision[i]:.2f})", color=colors[i])
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_pred, average="micro")

    ax.plot(recall["micro"], precision["micro"], label=f"Micro-average (AP={average_precision['micro']:.2f})", linestyle='--', color='black')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def roc_auc_plot(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], path: str = "reports/roc_auc.png") -> None:
    """Plot ROC AUC curves for each class.

    Args:
        y_true (np.ndarray): One-hot encoded true labels.
        y_pred (np.ndarray): Predicted probabilities for each class.
        labels (list[str]): List of class labels.
        path (str, optional): Path to save the plot. Defaults to "reports/roc_auc.png".
        
    Raises:
        ValueError: If the input arrays or labels are invalid.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("Invalid input: y_true and y_pred must be numpy arrays.")

    if not y_true.ndim == 2 or not y_pred.ndim == 2:
        raise ValueError("Invalid input: y_true and y_pred must be 2D numpy arrays.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Incompatible input shapes: y_true and y_pred must have the same number of samples.")

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError("Incompatible input shapes: y_true and y_pred must have the same number of classes.")
    
    if not labels or not isinstance(labels, list):
        raise ValueError("Invalid labels provided.")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("flare", len(labels))
    
    # Plot ROC curves for each class
    for i in range(len(labels)):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        display = RocCurveDisplay.from_predictions(
            y_true[:, i], y_pred[:, i],
            name=f"{labels[i]} (AUC={auc:.2f})",
            ax=ax, color=colors[i]
        )

    micro_auc = roc_auc_score(y_true, y_pred, average="micro", multi_class="ovr")
    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    
    ax.plot(fpr_micro, tpr_micro, 'k--', linewidth=2,
            label=f'Micro-average (AUC={micro_auc:.2f})')

    # Plot random guessing line
    ax.plot([0, 1], [0, 1], 'k:', label='Random Guessing (AUC=0.50)')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def report_plot(
    report: dict,
    labels: list[str],
    dir: str = "reports/"
) -> None:
    """
    Saves classification report(s) as plots.

    Args:
        report (dict): Classification report.
        labels (list[str]): List of class labels.
        dir (str): Directory to save the report.

    Raises:
        ValueError: If the report or labels are invalid.
    """
    if not isinstance(report, dict) or not report:
        raise ValueError("Invalid classification report provided.")
    
    if not isinstance(labels, list) or not labels:
        raise ValueError("Invalid labels provided.")

    os.makedirs(dir, exist_ok=True)

    with open(dir + "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    colors = sns.color_palette("flare", len(labels) + 2)
    
    metrics = ["precision", "recall", "f1-score", "support", "specificity"]
    data = {label: report[label] for label in labels if label in report}
    
    for metric in metrics: 
        plt.figure(figsize=(10, 6))
        scores = [data[label][metric] for label in labels if label in data]
        scores.append(report["macro avg"][metric])
        scores.append(report["weighted avg"][metric])
        
        df = pd.DataFrame({
            'Class': labels + ["Macro Avg", "Weighted Avg"],
            'Score': scores
        })
        ax = sns.barplot(x='Class', y='Score', data=df, hue='Class', palette=colors, legend=False)
        for container in ax.containers:
            ax.bar_label(container,  label_type="center", color="white")
        plt.title(f"Classification Report - {metric}")
        plt.xlabel("Class / Overall")
        plt.xticks(rotation=45)
        plt.ylabel(metric.title())
        plt.ylim(0, 1) if metric != "support" else plt.ylim(0, max(scores) * 1.1)
        plt.tight_layout()
        
        plt.savefig(dir + f"classification_report_{metric}.png", bbox_inches='tight')
        plt.close()

def report_add_specificity(report: dict, cm: np.ndarray, labels: list[str]) -> dict:
    """Adds specificity to the classification report.

    Args:
        report (dict): Classification report.
        cm (np.ndarray): Confusion matrix.
        labels (list[str]): List of class labels.

    Returns:
        dict: Updated classification report with specificity.
    
    Raises:
        ValueError: If the report, confusion matrix, or labels are invalid.
    """
    if report is None or not isinstance(report, dict):
        raise ValueError("Invalid classification report provided.")

    if not isinstance(cm, np.ndarray) or cm.ndim != 2:
        raise ValueError("Confusion matrix must be a 2D numpy array.")

    if not labels or not isinstance(labels, list):
        raise ValueError("Invalid labels provided.")

    supports = np.array([report[label]['support'] for label in labels])
    total_support = supports.sum()
    col_sums = cm.sum(axis=0)
    
    # Vectorized calcs of TP, FP, FN
    TP = np.diag(cm)
    FP = col_sums - TP
    TN = total_support - supports - FP
    
    # Calculate specificity
    denominators = TN + FP
    specificities = np.divide(TN, denominators, out=np.zeros_like(TN), where=denominators!=0)
    
    # Calculate non-weighted average
    report["macro avg"]["specificity"] = round(specificities.mean(), 4)
    # Calculate weighted average 
    weighted_spec = np.sum(specificities * supports) / total_support

    # Add values to report
    for i, label in enumerate(labels):
        report[label]['specificity'] = round(specificities[i], 4)

    report["weighted avg"]["specificity"] = round(weighted_spec, 4)
    
    return report

def save_metrics(gt: np.ndarray, preds: np.ndarray, labels: list[str], dir: str = "reports/") -> None:
    """Computes and saves all reports and plots for the passed true and predicted labels.
    
    Args:
        gt (np.ndarray): One-hot encoded ground truth labels.
        preds (np.ndarray): One-hot encoded predicted labels.
        labels (list[str]): List of class labels.
        dir (str, optional): Directory to save the reports and plots. Defaults to "reports/".

    Raises:
        ValueError: If the ground truth or predicted labels are empty.
    """
    os.makedirs(dir, exist_ok=True)

    if not gt.size or not preds.size:
        raise ValueError("Ground truth labels or predicted labels are empty.")

    # Convert predictions and ground truth to class labels
    y_pred = preds.argmax(axis=1)
    y_true = gt.argmax(axis=1)

    # Generate and plot PR and ROC curves
    precision_recall_plot(gt, preds, labels, dir + "precision_recall.png")
    roc_auc_plot(gt, preds, labels, dir + "roc_auc.png")

    # Create and plot confusion matrix
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    cm = confusion_matrix(y_true, y_pred, normalize=None)
    cm_plot(cm_norm, labels, title="Confusion Matrix", path=dir + "confusion_matrix.png")
    
    # Create and save classification report with specificity
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    assert isinstance(report, dict), "Classification report should be a dictionary."
    report = report_add_specificity(report, cm, labels)
    
    report_plot(report, labels, dir)

def feat_attention_plot(avg_attention: np.ndarray, feats: list[str], path: str = "reports/feature_attention.png") -> None:
    """Plot average feature attention weights.

    Args:
        avg_attention (np.ndarray): Average attention weights for features.
        feats (list[str]): List of feature names.
        path (str, optional): Path to save the plot. Defaults to "reports/feature_attention.png".
    """
    all_feats = feats + [f"mean_{i}" for i in feats] + [f"std_{i}" for i in feats]
    colors = sns.color_palette("flare", len(all_feats))
    
    plt.figure(figsize=(10, 4))
    plt.bar(all_feats, avg_attention, color=colors)
    plt.xlabel("Feature")
    plt.ylabel("Attention Weight")
    plt.title("Average Feature Attention Weights")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()