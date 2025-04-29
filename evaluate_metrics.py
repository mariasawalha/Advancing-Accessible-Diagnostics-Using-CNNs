import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

def evaluate_classification(y_test, y_pred, y_pred_probs, class_names=None):
    """
    Evaluate classification performance with various metrics and visualizations.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_probs: Predicted class probabilities
        class_names: List of class names (optional)
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
    
    # --- 1. Classification Report ---
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- 2. Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # --- 3. AUC-ROC Scores ---
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    auc_scores = roc_auc_score(y_test_bin, y_pred_probs, average=None)

    print("\n AUC-ROC per class:")
    for i, score in enumerate(auc_scores):
        print(f"  {class_names[i]}: {score:.4f}")

    # --- 4. ROC Curve Plot ---
    plt.figure(figsize=(10, 6))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_scores[i]:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    class_names = ["AMD", "DiabeticRetinopathy", "Glaucoma", "Normal"]
    print("This module is meant to be imported and used with data.")
    print("Example usage:")
    print("from evaluate_metrics import evaluate_classification")
    print("evaluate_classification(y_test, y_pred, y_pred_probs, class_names)")
