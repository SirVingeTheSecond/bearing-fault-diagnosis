"""
Configuration for CWRU Bearing Fault Diagnosis.
"""

import torch

config = {
    # Dataset - CWRU Bearing Dataset (1772 RPM / Load 1)
    # Source: https://github.com/srigas/CWRU_Bearing_NumPy
    "sampling_rate": 12000,  # 12 kHz drive end accelerometer
    "window_size": 2048,     # Samples per window (matches Rosa et al.)
    "stride": 64,            # ~97% overlap (per Rosa et al. 2024)

    # Classes - 10 bearing conditions
    "num_classes": 10,
    "class_names": [
        "Normal",
        "Ball_007", "Ball_014", "Ball_021",
        "IR_007", "IR_014", "IR_021",
        "OR_007", "OR_014", "OR_021",
    ],

    # Training
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "seed": 42,

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Early stopping
    "early_stopping": {
        "enabled": True,
        "patience": 10,
        "min_delta": 0.001,
    },

    # Split strategy: "random" or "fault_size"
    # - random: standard split (has data leakage)
    # - fault_size: train on 007+021, test on 014 (no leakage)
    "split_strategy": "fault_size",
    "test_fault_size": "014",
    "val_ratio": 0.2,
}


if __name__ == "__main__":
    for key, value in config.items():
        print(f"{key}: {value}")
