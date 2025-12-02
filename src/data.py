"""
CWRU Bearing Dataset loading and preprocessing.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from config import config

# 12kHz Drive End, Load 1 (1772 RPM)
# Format: (filename, class_name, fault_size)
CWRU_FILES = [
    ("1772_Normal.npz", "Normal", None),
    ("1772_B_7_DE12.npz", "Ball_007", "007"),
    ("1772_B_14_DE12.npz", "Ball_014", "014"),
    ("1772_B_21_DE12.npz", "Ball_021", "021"),
    ("1772_IR_7_DE12.npz", "IR_007", "007"),
    ("1772_IR_14_DE12.npz", "IR_014", "014"),
    ("1772_IR_21_DE12.npz", "IR_021", "021"),
    ("1772_OR@6_7_DE12.npz", "OR_007", "007"),
    ("1772_OR@6_14_DE12.npz", "OR_014", "014"),
    ("1772_OR@6_21_DE12.npz", "OR_021", "021"),
]


class CWRUDataset(Dataset):
    """PyTorch Dataset for CWRU bearing data."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


def load_raw_signals(data_dir: str = "data/raw") -> dict:
    """Load raw signals from .npz files."""
    class_names = config["class_names"]
    signals = {}

    for filename, class_name, fault_size in CWRU_FILES:
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing {filepath}")

        data = np.load(filepath)
        signal = data["DE"].flatten()

        signals[class_name] = {
            "signal": signal,
            "fault_size": fault_size if fault_size else "none",
            "label_idx": class_names.index(class_name),
        }

    return signals


def extract_windows(signals: dict, window_size: int, stride: int) -> tuple:
    """Extract fixed-length windows from raw signals."""
    X_list, y_list, fault_sizes = [], [], []

    for class_name, data in signals.items():
        signal = data["signal"]
        num_windows = (len(signal) - window_size) // stride + 1

        for i in range(num_windows):
            start = i * stride
            window = signal[start:start + window_size]
            X_list.append(window)
            y_list.append(data["label_idx"])
            fault_sizes.append(data["fault_size"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    fault_sizes = np.array(fault_sizes)

    # Normalize per sample
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    X = (X - mean) / std

    # Add channel dimension: (N, window_size) -> (N, 1, window_size)
    X = X[:, np.newaxis, :]

    return X, y, fault_sizes


def split_data(X: np.ndarray, y: np.ndarray, fault_sizes: np.ndarray, strategy: str) -> dict:
    """Split data using specified strategy."""
    seed = config["seed"]
    val_ratio = config["val_ratio"]

    if strategy == "fault_size":
        test_mask = fault_sizes == config["test_fault_size"]
        X_test, y_test = X[test_mask], y[test_mask]
        X_train_val, y_train_val = X[~test_mask], y[~test_mask]
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=seed, stratify=y_train_val
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def create_dataloaders(data: dict, batch_size: int) -> tuple:
    """Create train, val, test DataLoaders."""
    train_loader = DataLoader(
        CWRUDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        CWRUDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        CWRUDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_loader, val_loader, test_loader


def load_data(strategy: str = None, data_dir: str = "data/raw") -> dict:
    """Main entry point for loading CWRU data."""
    if strategy is None:
        strategy = config["split_strategy"]

    signals = load_raw_signals(data_dir)
    X, y, fault_sizes = extract_windows(
        signals, config["window_size"], config["stride"]
    )
    data = split_data(X, y, fault_sizes, strategy)

    print(f"Split: {strategy}")
    print(f"  Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")

    return data


if __name__ == "__main__":
    load_data(strategy="fault_size")
    load_data(strategy="random")