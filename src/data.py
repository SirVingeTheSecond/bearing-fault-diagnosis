"""
CWRU Bearing Dataset - with proper split to avoid segment-level leakage.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from config import config

CWRU_FILES = [
    ("1772_Normal.npz", "Normal", None, 0),
    ("1772_B_7_DE12.npz", "Ball_007", "007", 1),
    ("1772_B_14_DE12.npz", "Ball_014", "014", 1),
    ("1772_B_21_DE12.npz", "Ball_021", "021", 1),
    ("1772_IR_7_DE12.npz", "IR_007", "007", 2),
    ("1772_IR_14_DE12.npz", "IR_014", "014", 2),
    ("1772_IR_21_DE12.npz", "IR_021", "021", 2),
    ("1772_OR@6_7_DE12.npz", "OR_007", "007", 3),
    ("1772_OR@6_14_DE12.npz", "OR_014", "014", 3),
    ("1772_OR@6_21_DE12.npz", "OR_021", "021", 3),
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
    signals = {}
    mode = config["classification_mode"]

    for filename, class_name_10, fault_size, class_idx_4 in CWRU_FILES:
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing {filepath}")

        data = np.load(filepath)
        signal = data["DE"].flatten()

        if mode == "4class":
            label_idx = class_idx_4
        else:
            label_idx = config["class_names_10"].index(class_name_10)

        signals[class_name_10] = {
            "signal": signal,
            "fault_size": fault_size if fault_size else "none",
            "label_idx": label_idx,
        }

    return signals


def extract_windows_from_signal(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Extract windows from a single signal."""
    num_windows = (len(signal) - window_size) // stride + 1
    windows = np.zeros((num_windows, window_size), dtype=np.float32)

    for i in range(num_windows):
        start = i * stride
        window = signal[start:start + window_size]
        # Normalize per window
        mean = window.mean()
        std = window.std() + 1e-8
        windows[i] = (window - mean) / std

    return windows


def split_signal(signal: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> dict:
    """Split a raw signal into train/val/test segments (no overlap between splits)."""
    n = len(signal)
    
    # Calculate split points
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_size = n - val_size - test_size
    
    # Use random state for reproducibility
    rng = np.random.RandomState(seed)
    
    # Shuffle segment order (split into 10 chunks, shuffle, reassign)
    n_chunks = 10
    chunk_size = n // n_chunks
    chunks = [signal[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
    rng.shuffle(chunks)
    
    # Assign chunks to splits
    train_chunks = chunks[:6]  # 60%
    val_chunks = chunks[6:8]   # 20%
    test_chunks = chunks[8:]   # 20%
    
    return {
        "train": np.concatenate(train_chunks) if train_chunks else np.array([]),
        "val": np.concatenate(val_chunks) if val_chunks else np.array([]),
        "test": np.concatenate(test_chunks) if test_chunks else np.array([]),
    }


def load_data(strategy: str = None, data_dir: str = "data/raw") -> dict:
    """Load data with proper splitting to avoid segment-level leakage."""
    if strategy is None:
        strategy = config["split_strategy"]

    signals = load_raw_signals(data_dir)
    window_size = config["window_size"]
    stride = config["stride"]
    seed = config["seed"]

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for class_name, data in signals.items():
        signal = data["signal"]
        label = data["label_idx"]
        fault_size = data["fault_size"]

        if strategy == "fault_size":
            # Fault-size split: 014 goes entirely to test
            if fault_size == config["test_fault_size"]:
                windows = extract_windows_from_signal(signal, window_size, stride)
                X_test.append(windows)
                y_test.extend([label] * len(windows))
            else:
                # Split this signal into train/val (no test)
                split = split_signal(signal, val_ratio=0.2, test_ratio=0.0, seed=seed)
                
                train_windows = extract_windows_from_signal(split["train"], window_size, stride)
                val_windows = extract_windows_from_signal(split["val"], window_size, stride)
                
                X_train.append(train_windows)
                y_train.extend([label] * len(train_windows))
                X_val.append(val_windows)
                y_val.extend([label] * len(val_windows))

        else:  # random split
            # Split raw signal first, then extract windows
            split = split_signal(signal, val_ratio=0.2, test_ratio=0.2, seed=seed)
            
            train_windows = extract_windows_from_signal(split["train"], window_size, stride)
            val_windows = extract_windows_from_signal(split["val"], window_size, stride)
            test_windows = extract_windows_from_signal(split["test"], window_size, stride)
            
            X_train.append(train_windows)
            y_train.extend([label] * len(train_windows))
            X_val.append(val_windows)
            y_val.extend([label] * len(val_windows))
            X_test.append(test_windows)
            y_test.extend([label] * len(test_windows))

    # Concatenate and add channel dimension
    X_train = np.concatenate(X_train)[:, np.newaxis, :]
    X_val = np.concatenate(X_val)[:, np.newaxis, :]
    X_test = np.concatenate(X_test)[:, np.newaxis, :] if X_test else np.array([])

    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    print(f"Mode: {config['classification_mode']}, Split: {strategy}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def create_dataloaders(data: dict, batch_size: int) -> tuple:
    """Create train, val, test DataLoaders."""
    train_loader = DataLoader(
        CWRUDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        CWRUDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        CWRUDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=== 4-Class, Fault-Size Split ===")
    data = load_data(strategy="fault_size")
    print(f"  Train labels: {np.unique(data['y_train'])}")
    print(f"  Test labels:  {np.unique(data['y_test'])}")

    print("\n=== 4-Class, Random Split ===")
    data = load_data(strategy="random")
    print(f"  Train labels: {np.unique(data['y_train'])}")
    print(f"  Test labels:  {np.unique(data['y_test'])}")
