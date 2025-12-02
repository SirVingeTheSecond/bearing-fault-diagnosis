"""
Configuration for CWRU Bearing Fault Diagnosis.
"""

import torch

config = {
    # Dataset
    "sampling_rate": 12000,
    "window_size": 2048,
    "stride": 64,

    # Classification
    "classification_mode": "4class",
    "class_names_10": [
        "Normal",
        "Ball_007", "Ball_014", "Ball_021",
        "IR_007", "IR_014", "IR_021",
        "OR_007", "OR_014", "OR_021",
    ],
    "class_names_4": ["Normal", "Ball", "IR", "OR"],

    # Training
    "batch_size": 64,
    "epochs": 50,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "seed": 42,

    # Per-model learning rates
    "lr": {
        "cnn1d": 1e-3,
        "lstm": 1e-2,
        "cnnlstm": 1e-3,
    },

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Early stopping
    "early_stopping": {
        "enabled": True,
        "patience": 10,
        "min_delta": 0.001,
    },

    # Split
    "split_strategy": "fault_size",
    "test_fault_size": "021",
    "val_ratio": 0.2,
}

config["num_classes"] = 4 if config["classification_mode"] == "4class" else 10
config["class_names"] = config["class_names_4"] if config["classification_mode"] == "4class" else config["class_names_10"]


if __name__ == "__main__":
    for key, value in config.items():
        print(f"{key}: {value}")
