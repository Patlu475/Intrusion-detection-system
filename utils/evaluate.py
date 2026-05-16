"""Evaluation metrics and comparison utilities."""

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

DEFAULT_CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


def compute_metrics(y_true, y_pred, class_names=None):
    class_names = class_names or DEFAULT_CLASS_NAMES
    num_classes = len(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes),
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
        'per_class': {
            name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                    'f1': float(f1[i]), 'support': int(support[i])}
            for i, name in enumerate(class_names)
        },
        'classification_report': classification_report(
            y_true, y_pred, target_names=class_names, digits=4,
        ),
    }


def compute_confusion_matrix(y_true, y_pred, num_classes=None):
    n = num_classes or len(DEFAULT_CLASS_NAMES)
    return confusion_matrix(y_true, y_pred, labels=range(n))


def compute_roc_data(y_true, y_proba, class_names=None):
    class_names = class_names or DEFAULT_CLASS_NAMES
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(num_classes))
    roc_data = {}

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc),
        }

    all_fpr = np.unique(np.concatenate(
        [np.array(roc_data[n]['fpr']) for n in class_names]
    ))
    mean_tpr = np.zeros_like(all_fpr)
    for name in class_names:
        mean_tpr += np.interp(all_fpr, roc_data[name]['fpr'], roc_data[name]['tpr'])
    mean_tpr /= num_classes
    roc_data['macro'] = {
        'fpr': all_fpr.tolist(),
        'tpr': mean_tpr.tolist(),
        'auc': float(auc(all_fpr, mean_tpr)),
    }
    return roc_data


def format_comparison_table(results, class_names=None):
    class_names = class_names or DEFAULT_CLASS_NAMES
    header = f"{'Model':<20} {'Accuracy':>10} {'Macro F1':>10}"
    for name in class_names:
        header += f" {name+' F1':>10}"
    lines = [header, '-' * len(header)]

    for model_name, metrics in results.items():
        line = f"{model_name:<20} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}"
        for name in class_names:
            line += f" {metrics['per_class'][name]['f1']:>10.4f}"
        lines.append(line)

    return '\n'.join(lines)


def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
