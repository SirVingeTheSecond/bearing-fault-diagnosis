"""
Deep learning models for bearing fault diagnosis.
"""

import torch
import torch.nn as nn

from config import config


class CNN1D(nn.Module):
    """1D CNN for raw vibration signals."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class LSTM(nn.Module):
    """LSTM with convolutional downsampling for raw vibration signals."""

    def __init__(self, num_classes: int = 10, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # Downsample: 2048 -> ~128 timesteps
        self.downsample = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.downsample(x)  # (batch, 32, ~128)
        x = x.transpose(1, 2)   # (batch, ~128, 32)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)


class CNNLSTM(nn.Module):
    """CNN feature extraction + LSTM temporal modeling."""

    def __init__(self, num_classes: int = 10, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4, padding=30),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)


def get_model(name: str) -> nn.Module:
    """Factory function to get model by name."""
    models = {
        "cnn1d": CNN1D,
        "lstm": LSTM,
        "cnnlstm": CNNLSTM,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")

    return models[name](
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    )


if __name__ == "__main__":
    batch = torch.randn(4, 1, 2048)

    for name in ["cnn1d", "lstm", "cnnlstm"]:
        model = get_model(name)
        out = model(batch)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:8} | output: {out.shape} | params: {params:,}")
