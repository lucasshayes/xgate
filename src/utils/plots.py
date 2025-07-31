import os
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def cm_plot(
    cm: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    dir: str = "reports/confusion_matrix.png",
) -> None:
    """
    Plot a confusion matrix with optional normalization.

    Args:
        cm (np.ndarray): Confusion matrix.
        labels (list[str]): List of class labels.
        title (str): Title of the plot.
    """
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(dir + "confusion_matrix.png")

def report_plot(
    report: dict,
    labels: list[str],
    dir: str = "reports/classification_report.txt",
    colours: list[str] = ["#FF6940", "#F06068", "#DF5699", "#DF5699", "#60935D", "#93D28F"]
) -> None:
    """
    Save a classification report to a text file.

    Args:
        report (dict): Classification report.
        labels (list[str]): List of class labels.
        dir (str): Directory to save the report.
        colours (list[str]): List of colors for the plot.
    """
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    
    metrics = ["precision", "recall", "f1-score", "support"]
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
        sns.barplot(x='Class', y='Score', data=df, hue='Class', palette=colours, legend=False)
        plt.title(f"Classification Report - {metric}")
        plt.xlabel("Class / Overall")
        plt.xticks(rotation=45)
        plt.ylabel(metric.title())
        plt.ylim(0, 1) if metric != "support" else plt.ylim(0, max(scores) * 1.1)
        plt.tight_layout()
        plt.grid(True, alpha=0.25)
        
        plt.savefig(dir + f"classification_report_{metric}.png", bbox_inches='tight')
        plt.close()
