import torch

config = {
    # Dataset - CWRU Bearing Dataset
    "dataset_url": "https://engineering.case.edu/bearingdatacenter/download-data-file",
    "sampling_rate": 12000,  # 12 kHz drive end accelerometer
    "window_size": 2048,     # Samples per window (matching literature)
    "stride": 512,           # Window stride for overlapping segments

    # Classes - 10 bearing states
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
    # fault_size: train on 007+021, test on 014 (rigorous, no leakage)
    "split_strategy": "fault_size",
    "test_fault_size": "014",
    "val_ratio": 0.2,  # Validation split from training data
}


def print_config():
    print("=" * 50)
    print("Configuration")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
