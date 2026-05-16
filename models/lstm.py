"""Bidirectional LSTM with attention for intrusion detection."""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class Attention(nn.Module):
    """Bahdanau-style additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        energy = torch.tanh(self.attn_weights(lstm_output))
        scores = self.context_vector(energy).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attn_weights


class IDS_BiLSTM(nn.Module):
    def __init__(self, num_features=122, num_classes=5, seq_len=16, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.padded_dim = ((num_features + seq_len - 1) // seq_len) * seq_len
        self.input_dim = self.padded_dim // seq_len

        self.lstm1 = nn.LSTM(
            input_size=self.input_dim, hidden_size=64,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=128, hidden_size=32,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim=64)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        padded = torch.zeros(batch_size, self.padded_dim, device=x.device)
        padded[:, :x.size(1)] = x
        x = padded.view(batch_size, self.seq_len, self.input_dim)

        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)

        context, attn_weights = self.attention(x)
        logits = self.classifier(context)
        return logits, attn_weights


def build_model(num_features=122, num_classes=5, seq_len=16):
    return IDS_BiLSTM(num_features, num_classes, seq_len)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(X_batch)
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
            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return total_loss / len(dataloader.dataset), preds, labels


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001,
                device='cpu', patience=10, class_weights=None, checkpoint_path=None):
    model = model.to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
    }
    best_val_f1 = 0
    best_state = None
    no_improve = 0
    start_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        history = ckpt['history']
        best_val_f1 = ckpt['best_val_f1']
        best_state = ckpt['best_state']
        no_improve = ckpt['no_improve']
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} (best val F1: {best_val_f1:.4f})")

    for epoch in range(start_epoch, num_epochs):
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

        if checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'history': history,
                'best_val_f1': best_val_f1,
                'best_state': best_state,
                'no_improve': no_improve,
            }, checkpoint_path)

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history['best_epoch'] = len(history['val_f1']) - no_improve
    return history


def predict(model, X, device='cpu'):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        logits, _ = model(X_t)
        return logits.argmax(dim=1).cpu().numpy()


def predict_proba(model, X, device='cpu'):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        logits, _ = model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy()


def get_attention_weights(model, X, device='cpu'):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        _, attn = model(X_t)
        return attn.cpu().numpy()
