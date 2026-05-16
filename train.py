"""Unified training script for all IDS models."""

import argparse
import json
import os
import random

import joblib
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from utils.preprocess import get_data, get_unsw_data, get_dataloaders
from utils.evaluate import compute_metrics, save_metrics
from models.random_forest import train as train_rf, train_with_gridsearch, predict, predict_proba
from models.cnn import build_model as build_cnn, train_model as train_cnn_model
from models.cnn import predict as cnn_predict, predict_proba as cnn_predict_proba
from models.lstm import build_model as build_lstm, train_model as train_lstm_model
from models.lstm import predict as lstm_predict, predict_proba as lstm_predict_proba


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_class_weights(original_counts):
    classes = np.arange(len(original_counts))
    y_fake = np.repeat(classes, original_counts)
    weights = compute_class_weight('balanced', classes=classes, y=y_fake)
    return weights.astype(np.float32)


def do_train_rf(data, output_dir, quick=False):
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)

    if quick:
        model = train_rf(data['X_train'], data['y_train'])
    else:
        model, best_params = train_with_gridsearch(
            data['X_train'], data['y_train'],
        )
        with open(os.path.join(output_dir, 'rf_best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)

    y_pred = predict(model, data['X_test'])
    y_proba = predict_proba(model, data['X_test'])
    metrics = compute_metrics(data['y_test'], y_pred, data['class_names'])

    print(f"\nRandom Forest Results:")
    print(metrics['classification_report'])

    joblib.dump(model, os.path.join(output_dir, 'rf_model.joblib'))
    save_metrics(metrics, os.path.join(output_dir, 'rf_metrics.json'))
    np.save(os.path.join(output_dir, 'rf_proba.npy'), y_proba)

    return metrics


def do_train_cnn(data, output_dir, device, epochs=50, batch_size=256, lr=0.001):
    print("\n" + "="*60)
    print("Training 1D CNN")
    print("="*60)

    train_loader, val_loader = get_dataloaders(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        batch_size=batch_size,
    )

    model = build_cnn(data['num_features'], data['num_classes'])
    class_weights = compute_class_weights(data['original_class_counts'])
    print(f"Class weights: {class_weights}")

    history = train_cnn_model(
        model, train_loader, val_loader,
        num_epochs=epochs, lr=lr, device=str(device),
        patience=10, class_weights=class_weights,
    )

    y_pred = cnn_predict(model, data['X_test'], device=str(device))
    y_proba = cnn_predict_proba(model, data['X_test'], device=str(device))
    metrics = compute_metrics(data['y_test'], y_pred, data['class_names'])

    print(f"\n1D CNN Results:")
    print(metrics['classification_report'])

    torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_model.pt'))
    save_metrics(metrics, os.path.join(output_dir, 'cnn_metrics.json'))
    with open(os.path.join(output_dir, 'cnn_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    np.save(os.path.join(output_dir, 'cnn_proba.npy'), y_proba)

    return metrics


def do_train_lstm(data, output_dir, device, epochs=50, batch_size=256, lr=0.001):
    print("\n" + "="*60)
    print("Training BiLSTM with Attention")
    print("="*60)

    train_loader, val_loader = get_dataloaders(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        batch_size=batch_size,
    )

    model = build_lstm(data['num_features'], data['num_classes'])
    class_weights = compute_class_weights(data['original_class_counts'])

    history = train_lstm_model(
        model, train_loader, val_loader,
        num_epochs=epochs, lr=lr, device=str(device),
        patience=10, class_weights=class_weights,
    )

    y_pred = lstm_predict(model, data['X_test'], device=str(device))
    y_proba = lstm_predict_proba(model, data['X_test'], device=str(device))
    metrics = compute_metrics(data['y_test'], y_pred, data['class_names'])

    print(f"\nBiLSTM-Attention Results:")
    print(metrics['classification_report'])

    torch.save(model.state_dict(), os.path.join(output_dir, 'lstm_model.pt'))
    save_metrics(metrics, os.path.join(output_dir, 'lstm_metrics.json'))
    with open(os.path.join(output_dir, 'lstm_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    np.save(os.path.join(output_dir, 'lstm_proba.npy'), y_proba)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train IDS models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['rf', 'cnn', 'lstm', 'all'])
    parser.add_argument('--dataset', type=str, default='nsl-kdd',
                        choices=['nsl-kdd', 'unsw-nb15'])
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--no-smote', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Skip grid search for RF')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Output dir: {output_dir}")

    if args.dataset == 'unsw-nb15':
        data = get_unsw_data(args.data_dir, use_smote=not args.no_smote)
    else:
        data = get_data(args.data_dir, use_smote=not args.no_smote)

    results = {}

    if args.model in ('rf', 'all'):
        results['Random Forest'] = do_train_rf(
            data, output_dir, quick=args.quick,
        )

    if args.model in ('cnn', 'all'):
        results['1D CNN'] = do_train_cnn(
            data, output_dir, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )

    if args.model in ('lstm', 'all'):
        results['BiLSTM-Attention'] = do_train_lstm(
            data, output_dir, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )

    if len(results) > 1:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        from utils.evaluate import format_comparison_table
        print(format_comparison_table(results, data['class_names']))


if __name__ == '__main__':
    main()
