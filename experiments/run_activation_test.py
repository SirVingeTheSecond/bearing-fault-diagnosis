"""
Activation function ablation experiment.

Tests: ReLU, Leaky ReLU, GELU, ELU, SELU on CNN with fault-size split.

Usage:
    python experiments/run_activation_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.data import load_data, create_dataloaders
from src.models import create_model
from src.training import train
from src.evaluation import evaluate
from src.utils import set_seed, get_device, print_header, print_separator

ACTIVATIONS = ["relu", "leaky_relu", "gelu", "elu", "selu"]
SEEDS = [42, 123, 456]


def main():
    device = get_device()
    results = []

    print_header("ACTIVATION FUNCTION TEST")
    print(f"Model: CNN | Split: fault_size_all_loads | Seeds: {SEEDS}")
    print(f"Activations: {ACTIVATIONS}\n")

    # Load data once
    data = load_data(mode="4class", split="fault_size_all_loads", seed=42, verbose=False)

    for activation in ACTIVATIONS:
        accuracies = []

        for seed in SEEDS:
            set_seed(seed)

            model = create_model("cnn", num_classes=4, activation=activation).to(device)
            train_loader, val_loader, test_loader = create_dataloaders(data, "4class")

            train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader if len(val_loader.dataset) > 0 else None,
                mode="4class",
                epochs=config.EPOCHS,
                device=device,
                verbose=False,
            )

            metrics = evaluate(model, test_loader, device, "4class")
            accuracies.append(metrics["accuracy"])
            print(f"  {activation:12} seed={seed}: {metrics['accuracy']:.4f}")

        mean_acc = sum(accuracies) / len(accuracies)
        std_acc = (sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)) ** 0.5
        results.append((activation, mean_acc * 100, std_acc * 100))
        print(f"  {activation:12} MEAN: {mean_acc*100:.2f}% +- {std_acc*100:.2f}%\n")

    print_separator()
    print("SUMMARY")
    print_separator()
    print(f"{'Activation':<12} {'Accuracy (%)':>12} {'Std':>10}")
    print_separator(char="-", width=36)

    for activation, mean, std in sorted(results, key=lambda x: -x[1]):
        print(f"{activation:<12} {mean:>12.2f} {std:>10.2f}")

    best = max(results, key=lambda x: x[1])
    print(f"\nBest: {best[0]} ({best[1]:.2f}% +- {best[2]:.2f}%)")


if __name__ == "__main__":
    main()
