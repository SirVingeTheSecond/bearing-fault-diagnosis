"""
Training and evaluation functions.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import config


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: str) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: str) -> tuple:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)

        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = None, device: str = None, model_name: str = None) -> dict:
    """Full training loop with early stopping."""
    if epochs is None:
        epochs = config["epochs"]
    if device is None:
        device = config["device"]

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Get learning rate (per-model or default)
    if model_name and isinstance(config["lr"], dict):
        lr = config["lr"].get(model_name, 1e-3)
    elif isinstance(config["lr"], dict):
        lr = 1e-3  # default
    else:
        lr = config["lr"]

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])

    early_stopping = None
    if config["early_stopping"]["enabled"]:
        early_stopping = EarlyStopping(
            patience=config["early_stopping"]["patience"],
            min_delta=config["early_stopping"]["min_delta"],
        )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        if early_stopping and early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    return {"history": history, "best_val_acc": best_val_acc}


if __name__ == "__main__":
    from data import load_data, create_dataloaders
    from models import get_model

    data = load_data(strategy="random")
    train_loader, val_loader, _ = create_dataloaders(data, config["batch_size"])

    model = get_model("cnn1d")
    result = train_model(model, train_loader, val_loader, epochs=5, model_name="cnn1d")

    print(f"\nBest val accuracy: {result['best_val_acc']:.4f}")
