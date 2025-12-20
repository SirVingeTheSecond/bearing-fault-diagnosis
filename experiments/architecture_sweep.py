"""
Architecture sweep experiment.

Tests model variants and dropout values on fault-size split.

Usage:
    python experiments/architecture_sweep.py full
    python experiments/architecture_sweep.py dropout cnn
    python experiments/architecture_sweep.py quick
"""

import sys
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.data import load_data, create_dataloaders
from src.evaluation import evaluate
from src.models import create_model
from src.training import train
from src.utils import set_seed, get_device, count_parameters, print_header, print_separator

RESULTS_DIR = Path(config.RESULTS_DIR)
SEEDS = [42, 123, 456]


def run_single_experiment(
    model_name: str,
    dropout: float,
    mode: str = "4class",
    split: str = "fault_size_all_loads",
    seed: int = 42,
    epochs: int = 100,
    verbose: bool = True,
) -> dict:
    """Run a single architecture experiment."""
    set_seed(seed)
    device = get_device()

    data = load_data(mode=mode, split=split, seed=seed, verbose=False)
    train_loader, val_loader, test_loader = create_dataloaders(data, mode)

    num_classes = config.NUM_CLASSES[mode]
    model = create_model(model_name, num_classes=num_classes, dropout=dropout)
    model = model.to(device)
    params = count_parameters(model)

    if verbose:
        print(f"\n  Model: {model_name} | Dropout: {dropout} | Params: {params:,}")

    train_result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if len(val_loader.dataset) > 0 else None,
        mode=mode,
        epochs=epochs,
        device=device,
        verbose=verbose,
    )

    test_metrics = evaluate(model, test_loader, device, mode)

    per_class_f1 = {
        name: metrics["f1"]
        for name, metrics in test_metrics["per_class_metrics"].items()
    }

    if verbose:
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Per-class F1: Normal={per_class_f1.get('Normal', 0):.3f}, "
              f"Ball={per_class_f1.get('Ball', 0):.3f}, "
              f"IR={per_class_f1.get('IR', 0):.3f}, "
              f"OR={per_class_f1.get('OR', 0):.3f}")

    return {
        "model": model_name,
        "dropout": dropout,
        "mode": mode,
        "split": split,
        "seed": seed,
        "parameters": params,
        "epochs_trained": train_result["epochs_trained"],
        "best_val_acc": train_result["best_val_acc"],
        "test_accuracy": test_metrics["accuracy"],
        "per_class_f1": per_class_f1,
        "macro_f1": test_metrics["macro_f1"],
        "timestamp": datetime.now().isoformat(),
    }


def run_architecture_sweep(seeds: list = None) -> list:
    """
    Run architecture sweep.

    Tests:
    - Model variants: cnn, lstm, cnnlstm
    - Dropout values: 0.1, 0.2, 0.3, 0.4, 0.5
    """
    if seeds is None:
        seeds = SEEDS

    experiments = [
        ("cnn", [0.1, 0.2, 0.3, 0.4, 0.5]),
        ("lstm", [0.1, 0.2, 0.3, 0.4, 0.5]),
        ("cnnlstm", [0.1, 0.2, 0.3, 0.4, 0.5]),
    ]

    total = sum(len(dropouts) * len(seeds) for _, dropouts in experiments)
    current = 0

    print_header("ARCHITECTURE SWEEP")
    print(f"Total experiments: {total}")
    print(f"Seeds: {seeds}\n")

    all_results = []

    for model_name, dropout_values in experiments:
        for dropout in dropout_values:
            for seed in seeds:
                current += 1
                print(f"[{current}/{total}] {model_name} | dropout={dropout} | seed={seed}")

                result = run_single_experiment(
                    model_name=model_name,
                    dropout=dropout,
                    seed=seed,
                    verbose=False,
                )
                all_results.append(result)

                print(f"  -> Acc: {result['test_accuracy']:.4f}, F1: {result['macro_f1']:.4f}")

    return all_results


def run_dropout_sweep(model_name: str = "cnn", seeds: list = None) -> list:
    """Focused dropout sweep for a single model."""
    if seeds is None:
        seeds = SEEDS

    dropout_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    total = len(dropout_values) * len(seeds)
    current = 0

    print_header(f"DROPOUT SWEEP: {model_name}")
    print(f"Dropout values: {dropout_values}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {total}\n")

    all_results = []

    for dropout in dropout_values:
        for seed in seeds:
            current += 1
            print(f"[{current}/{total}] dropout={dropout} | seed={seed}")

            result = run_single_experiment(
                model_name=model_name,
                dropout=dropout,
                seed=seed,
                verbose=False,
            )
            all_results.append(result)

            print(f"  -> Acc: {result['test_accuracy']:.4f}, F1: {result['macro_f1']:.4f}")

    return all_results


def aggregate_results(results: list) -> dict:
    """Aggregate results by model and dropout, computing mean and std."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        key = (r["model"], r["dropout"])
        grouped[key].append(r)

    aggregated = {}
    for (model, dropout), runs in grouped.items():
        accuracies = [r["test_accuracy"] for r in runs]
        macro_f1s = [r["macro_f1"] for r in runs]
        ball_f1s = [r["per_class_f1"].get("Ball", 0) for r in runs]
        ir_f1s = [r["per_class_f1"].get("IR", 0) for r in runs]
        or_f1s = [r["per_class_f1"].get("OR", 0) for r in runs]

        aggregated[(model, dropout)] = {
            "model": model,
            "dropout": dropout,
            "n_runs": len(runs),
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "macro_f1_mean": np.mean(macro_f1s),
            "macro_f1_std": np.std(macro_f1s),
            "ball_f1_mean": np.mean(ball_f1s),
            "ball_f1_std": np.std(ball_f1s),
            "ir_f1_mean": np.mean(ir_f1s),
            "ir_f1_std": np.std(ir_f1s),
            "or_f1_mean": np.mean(or_f1s),
            "or_f1_std": np.std(or_f1s),
            "parameters": runs[0]["parameters"],
        }

    return aggregated


def print_summary(aggregated: dict):
    """Print formatted summary of aggregated results."""
    print_separator()
    print("ARCHITECTURE SWEEP SUMMARY")
    print_separator()

    print(f"\n{'Model':<12} {'Drop':<6} {'Params':<10} "
          f"{'Accuracy':<14} {'Macro F1':<14} {'Ball F1':<14} {'IR F1':<14}")
    print("-" * 90)

    sorted_keys = sorted(aggregated.keys(), key=lambda k: (k[0], k[1]))

    for key in sorted_keys:
        r = aggregated[key]
        print(f"{r['model']:<12} {r['dropout']:<6.1f} {r['parameters']:<10,} "
              f"{r['accuracy_mean']:.3f}+/-{r['accuracy_std']:.3f}  "
              f"{r['macro_f1_mean']:.3f}+/-{r['macro_f1_std']:.3f}  "
              f"{r['ball_f1_mean']:.3f}+/-{r['ball_f1_std']:.3f}  "
              f"{r['ir_f1_mean']:.3f}+/-{r['ir_f1_std']:.3f}")

    # Best configurations
    best_acc = max(aggregated.values(), key=lambda x: x["accuracy_mean"])
    best_f1 = max(aggregated.values(), key=lambda x: x["macro_f1_mean"])

    print("\n" + "-" * 90)
    print("BEST CONFIGURATIONS:")
    print(f"  Best Accuracy: {best_acc['model']} (dropout={best_acc['dropout']}) "
          f"-> {best_acc['accuracy_mean']:.3f}+/-{best_acc['accuracy_std']:.3f}")
    print(f"  Best Macro F1: {best_f1['model']} (dropout={best_f1['dropout']}) "
          f"-> {best_f1['macro_f1_mean']:.3f}+/-{best_f1['macro_f1_std']:.3f}")


def save_results(results: list, aggregated: dict, filename: str):
    """Save raw and aggregated results to JSON."""
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    output = {
        "raw_results": results,
        "aggregated": {f"{k[0]}_{k[1]}": v for k, v in aggregated.items()},
        "timestamp": datetime.now().isoformat(),
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\nResults saved to: {filename}")


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "full"

    if mode == "full":
        results = run_architecture_sweep(seeds=SEEDS)
        aggregated = aggregate_results(results)
        print_summary(aggregated)
        save_results(results, aggregated, str(RESULTS_DIR / "architecture_sweep.json"))

    elif mode == "dropout":
        model = sys.argv[2] if len(sys.argv) > 2 else "cnn"
        results = run_dropout_sweep(model_name=model, seeds=SEEDS)
        aggregated = aggregate_results(results)
        print_summary(aggregated)
        save_results(results, aggregated, str(RESULTS_DIR / f"dropout_sweep_{model}.json"))

    elif mode == "quick":
        results = run_architecture_sweep(seeds=[42])
        aggregated = aggregate_results(results)
        print_summary(aggregated)
        save_results(results, aggregated, str(RESULTS_DIR / "architecture_sweep_quick.json"))

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python architecture_sweep.py [full|dropout|quick]")
        print("  full          - Full sweep with all seeds")
        print("  dropout MODEL - Dropout sweep for specific model")
        print("  quick         - Quick test with single seed")


if __name__ == "__main__":
    main()
