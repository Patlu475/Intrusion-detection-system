"""1D Convolutional Neural Network for intrusion detection."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class IDS_CNN(nn.Module):
    def __init__(self, num_features=122, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_features=122, num_classes=5):
    return IDS_CNN(num_features, num_classes)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(dataloader.dataset)
    train_f1 = f1_score(labels, preds, average='macro')
    return avg_loss, train_f1


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / len(dataloader.dataset), preds, labels


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001,
                device='cpu', patience=10, class_weights=None):
    model = model.to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True,
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
    }
    best_val_f1 = 0
    best_state = None
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = evaluate_epoch(
            model, val_loader, criterion, device,
        )
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    history['best_epoch'] = len(history['val_f1']) - no_improve
    return history


def predict(model, X, device='cpu'):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        logits = model(X_t)
        return logits.argmax(dim=1).cpu().numpy()


def predict_proba(model, X, device='cpu'):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        logits = model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy()
