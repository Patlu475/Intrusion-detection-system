"""Load trained models and produce comparison on test set."""

import argparse
import os

import joblib
import numpy as np
import torch

from utils.preprocess import get_data, get_unsw_data
from utils.evaluate import (
    compute_metrics, compute_confusion_matrix, compute_roc_data,
    format_comparison_table, save_metrics,
)
from models.cnn import build_model as build_cnn, predict_proba as cnn_proba
from models.lstm import build_model as build_lstm, predict_proba as lstm_proba


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_models(output_dir, num_features, num_classes, device):
    models = {}

    rf_path = os.path.join(output_dir, 'rf_model.joblib')
    if os.path.exists(rf_path):
        models['Random Forest'] = joblib.load(rf_path)

    cnn_path = os.path.join(output_dir, 'cnn_model.pt')
    if os.path.exists(cnn_path):
        cnn = build_cnn(num_features, num_classes)
        cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        cnn.eval()
        models['1D CNN'] = cnn

    lstm_path = os.path.join(output_dir, 'lstm_model.pt')
    if os.path.exists(lstm_path):
        lstm = build_lstm(num_features, num_classes)
        lstm.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
        lstm.eval()
        models['BiLSTM-Attention'] = lstm

    return models


def evaluate_all(models, X_test, y_test, device, class_names):
    results = {}

    for name, model in models.items():
        if name == 'Random Forest':
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
        elif name == '1D CNN':
            y_proba = cnn_proba(model, X_test, device=str(device))
            y_pred = y_proba.argmax(axis=1)
        elif name == 'BiLSTM-Attention':
            y_proba = lstm_proba(model, X_test, device=str(device))
            y_pred = y_proba.argmax(axis=1)

        metrics = compute_metrics(y_test, y_pred, class_names)
        cm = compute_confusion_matrix(y_test, y_pred, len(class_names))
        roc = compute_roc_data(y_test, y_proba, class_names)

        results[name] = {
            **metrics,
            'confusion_matrix': cm.tolist(),
            'roc_data': roc,
            'y_pred': y_pred,
            'y_proba': y_proba,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare trained IDS models')
    parser.add_argument('--dataset', type=str, default='nsl-kdd',
                        choices=['nsl-kdd', 'unsw-nb15'])
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='output')
    args = parser.parse_args()

    device = get_device()
    output_dir = os.path.join(args.output_dir, args.dataset)

    if args.dataset == 'unsw-nb15':
        data = get_unsw_data(args.data_dir, use_smote=False)
    else:
        data = get_data(args.data_dir, use_smote=False)

    models = load_models(
        output_dir, data['num_features'], data['num_classes'], device,
    )
    print(f"Loaded models: {list(models.keys())}")

    results = evaluate_all(
        models, data['X_test'], data['y_test'], device, data['class_names'],
    )

    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON — {args.dataset.upper()}")
    print(f"{'='*60}")

    serializable = {}
    for name, r in results.items():
        print(f"\n--- {name} ---")
        print(r['classification_report'])
        serializable[name] = {
            k: v for k, v in r.items()
            if k not in ('y_pred', 'y_proba', 'classification_report')
        }

    print("\n" + format_comparison_table(results, data['class_names']))

    save_metrics(serializable, os.path.join(output_dir, 'comparison_results.json'))
    print(f"\nResults saved to {output_dir}/comparison_results.json")


if __name__ == '__main__':
    main()
