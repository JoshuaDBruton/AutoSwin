from sklearn.metrics import roc_auc_score
import numpy as np


def macro_auc(y_true, y_prob):
    """Unweighted mean across classes with both positive and negative examples."""
    scores = [roc_auc_score(y_true[:, i], y_prob[:, i])
              for i in range(y_true.shape[1]) if len(np.unique(y_true[:, i])) == 2]
    return float(np.mean(scores)) if scores else float("nan")


def calculate_per_class_metrics(confusion_matrix, prefix, y_true, y_prob, class_names=None):
    if len(confusion_matrix.shape) != 3:
        raise ValueError("Expected 3D confusion matrix for multilabel classification")

    n_classes = confusion_matrix.shape[0]

    if class_names is None:
        class_names = [f"Class_{i:02d}" for i in range(n_classes)]

    metrics_dict = {}
    total_auc = 0
    total_classes = 0

    for i, class_name in enumerate(class_names):
        cm = confusion_matrix[i]
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        # Calculate all metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as sensitivity
        sensitivity = recall  # Sensitivity = Recall = TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        support = tp + fn  # actual positives
        auc = (roc_auc_score(y_true[:, i], y_prob[:, i])
               if len(np.unique(y_true[:, i])) == 2 else float("nan"))

        # Log individual class metrics
        metrics_dict.update({
            f"{prefix}precision/{class_name}": precision,
            f"{prefix}sensitivity/{class_name}": sensitivity,
            f"{prefix}specificity/{class_name}": specificity,
            f"{prefix}f1/{class_name}": f1,
            f"{prefix}accuracy/{class_name}": accuracy,
            f"{prefix}auc/{class_name}": auc,
            f"{prefix}support/{class_name}": int(support),
            f"{prefix}tp/{class_name}": int(tp),
            f"{prefix}fp/{class_name}": int(fp),
            f"{prefix}fn/{class_name}": int(fn),
            f"{prefix}tn/{class_name}": int(tn)
        })

        if np.isfinite(auc):
            total_auc += auc
            total_classes += 1

    return metrics_dict, (total_auc / total_classes if total_classes else float("nan"))
