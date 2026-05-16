"""Generate publication-quality figures for the IDS comparison."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from utils.evaluate import load_metrics
from utils.preprocess import CLASS_NAMES as NSL_CLASS_NAMES, UNSW_CLASS_NAMES

MODEL_NAMES = ['Random Forest', '1D CNN', 'BiLSTM-Attention']
COLORS = ['#2196F3', '#FF9800', '#4CAF50']


def set_style():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_confusion_matrices(output_dir, figures_dir, class_names):
    comparison = load_metrics(os.path.join(output_dir, 'comparison_results.json'))

    available = [n for n in MODEL_NAMES if n in comparison]
    ncols = len(available)
    if ncols == 0:
        return

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for idx, name in enumerate(available):
        cm = np.array(comparison[name]['confusion_matrix'])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_norm[i, j]:.1%}\n({cm[i, j]})"

        sns.heatmap(
            cm_norm, annot=annot, fmt='', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[idx], vmin=0, vmax=1,
        )
        axes[idx].set_title(name, fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label' if idx == 0 else '')
        axes[idx].set_xlabel('Predicted Label')

    plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'confusion_matrices.png'))
    plt.savefig(os.path.join(figures_dir, 'confusion_matrices.pdf'))
    plt.close()
    print("  Saved confusion_matrices.png/pdf")


def plot_roc_curves(output_dir, figures_dir, class_names):
    comparison = load_metrics(os.path.join(output_dir, 'comparison_results.json'))

    classes_to_plot = class_names + ['macro']
    ncols = 3
    nrows = (len(classes_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for idx, cls in enumerate(classes_to_plot):
        ax = axes[idx]
        for model_name, color in zip(MODEL_NAMES, COLORS):
            if model_name not in comparison:
                continue
            roc = comparison[model_name]['roc_data'].get(cls)
            if not roc:
                continue
            ax.plot(
                roc['fpr'], roc['tpr'], color=color,
                label=f"{model_name} (AUC={roc['auc']:.3f})", linewidth=2,
            )
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        title = "Macro Average" if cls == 'macro' else cls
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    for idx in range(len(classes_to_plot), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'roc_curves.png'))
    plt.savefig(os.path.join(figures_dir, 'roc_curves.pdf'))
    plt.close()
    print("  Saved roc_curves.png/pdf")


def plot_f1_comparison(output_dir, figures_dir, class_names):
    comparison = load_metrics(os.path.join(output_dir, 'comparison_results.json'))

    categories = class_names + ['Macro Avg']
    x = np.arange(len(categories))
    available = [n for n in MODEL_NAMES if n in comparison]
    width = 0.8 / max(len(available), 1)

    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 1.5), 6))

    for i, name in enumerate(available):
        color = COLORS[MODEL_NAMES.index(name)]
        f1s = [comparison[name]['per_class'][c]['f1'] for c in class_names]
        f1s.append(comparison[name]['macro_f1'])

        bars = ax.bar(x + i * width, f1s, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(categories, fontsize=10, rotation=30, ha='right')
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'f1_comparison.png'))
    plt.savefig(os.path.join(figures_dir, 'f1_comparison.pdf'))
    plt.close()
    print("  Saved f1_comparison.png/pdf")


def plot_training_curves(output_dir, figures_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for col, (model_key, model_label) in enumerate([('cnn', '1D CNN'), ('lstm', 'BiLSTM-Attention')]):
        hist_path = os.path.join(output_dir, f'{model_key}_history.json')
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            hist = json.load(f)

        epochs = range(1, len(hist['train_loss']) + 1)
        best = hist.get('best_epoch', len(hist['train_loss']))

        axes[0, col].plot(epochs, hist['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, col].plot(epochs, hist['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, col].axvline(best, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {best})')
        axes[0, col].set_title(f'{model_label} — Loss', fontsize=13, fontweight='bold')
        axes[0, col].set_xlabel('Epoch')
        axes[0, col].set_ylabel('Loss')
        axes[0, col].legend()
        axes[0, col].grid(alpha=0.3)

        axes[1, col].plot(epochs, hist['train_f1'], 'b-', label='Train', linewidth=2)
        axes[1, col].plot(epochs, hist['val_f1'], 'r-', label='Val', linewidth=2)
        axes[1, col].axvline(best, color='green', linestyle='--', alpha=0.7, label=f'Best (epoch {best})')
        axes[1, col].set_title(f'{model_label} — Macro F1', fontsize=13, fontweight='bold')
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].set_ylabel('Macro F1')
        axes[1, col].legend()
        axes[1, col].grid(alpha=0.3)

    plt.suptitle('Training Curves', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_curves.png'))
    plt.savefig(os.path.join(figures_dir, 'training_curves.pdf'))
    plt.close()
    print("  Saved training_curves.png/pdf")


def plot_attention_heatmap(output_dir, figures_dir, dataset, data_dir='data'):
    from models.lstm import build_model as build_lstm, get_attention_weights

    lstm_path = os.path.join(output_dir, 'lstm_model.pt')
    if not os.path.exists(lstm_path):
        print("  Skipping attention heatmap (no LSTM model found)")
        return

    if dataset == 'unsw-nb15':
        from utils.preprocess import get_unsw_data
        data = get_unsw_data(data_dir, use_smote=False)
    else:
        from utils.preprocess import get_data
        data = get_data(data_dir, use_smote=False)

    model = build_lstm(data['num_features'], data['num_classes'])
    model.load_state_dict(torch.load(lstm_path, map_location='cpu', weights_only=True))
    model.eval()

    samples = []
    sample_labels = []
    for cls_idx, cls_name in enumerate(data['class_names']):
        mask = data['y_test'] == cls_idx
        if mask.any():
            idx = np.where(mask)[0][0]
            samples.append(data['X_test'][idx])
            sample_labels.append(cls_name)

    if not samples:
        return

    X_sample = np.stack(samples)
    attn = get_attention_weights(model, X_sample)

    fig, ax = plt.subplots(figsize=(10, max(3, len(samples) * 0.6)))
    sns.heatmap(
        attn, annot=True, fmt='.3f', cmap='YlOrRd',
        xticklabels=[f't{i}' for i in range(attn.shape[1])],
        yticklabels=sample_labels, ax=ax,
    )
    ax.set_title('Attention Weights by Class (BiLSTM)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Attack Class')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'attention_heatmap.png'))
    plt.savefig(os.path.join(figures_dir, 'attention_heatmap.pdf'))
    plt.close()
    print("  Saved attention_heatmap.png/pdf")


def main():
    parser = argparse.ArgumentParser(description='Generate IDS report figures')
    parser.add_argument('--dataset', type=str, default='nsl-kdd',
                        choices=['nsl-kdd', 'unsw-nb15'])
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--figures-dir', type=str, default='figures')
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.dataset)
    figures_dir = os.path.join(args.figures_dir, args.dataset)
    os.makedirs(figures_dir, exist_ok=True)
    set_style()

    if args.dataset == 'unsw-nb15':
        class_names = UNSW_CLASS_NAMES
    else:
        class_names = NSL_CLASS_NAMES

    print(f"Generating figures for {args.dataset.upper()}...")
    plot_confusion_matrices(output_dir, figures_dir, class_names)
    plot_roc_curves(output_dir, figures_dir, class_names)
    plot_f1_comparison(output_dir, figures_dir, class_names)
    plot_training_curves(output_dir, figures_dir)
    plot_attention_heatmap(output_dir, figures_dir, args.dataset, args.data_dir)
    print(f"Done! All figures saved to {figures_dir}")


if __name__ == '__main__':
    main()
