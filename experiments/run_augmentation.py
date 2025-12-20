"""
Data augmentation experiment.

Tests various augmentation policies on fault-size split.

Usage:
    python experiments/run_augmentation.py --mode quick
    python experiments/run_augmentation.py --mode full
    python experiments/run_augmentation.py --mode single --policy moderate
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from src import config
from src.models import create_model
from src.data import load_data, SignalDataset
from src.training import train
from src.evaluation import evaluate
from src.utils import get_device, set_seed, count_parameters, print_header, print_separator
from src.augmentation import AugmentedSignalDataset, get_augmentation_policy

RESULTS_DIR = Path(config.RESULTS_DIR) / "augmentation"

POLICIES = ["none", "noise_only", "warp_only", "scale_only", "light", "moderate", "heavy"]
SEEDS = [42, 123, 456]


def run_single_experiment(
    model_name: str = "cnn",
    policy_name: str = "none",
    mode: str = "4class",
    split: str = "fault_size_all_loads",
    seed: int = 42,
    epochs: int = 100,
    verbose: bool = True,
) -> dict:
    """Run single experiment with specified augmentation policy."""
    set_seed(seed)
    device = get_device()

    if verbose:
        print_header(f"Model: {model_name} | Aug: {policy_name} | Seed: {seed}", width=60)

    # Load data
    data = load_data(mode=mode, split=split, seed=seed, verbose=False)

    # Create datasets
    base_train_dataset = SignalDataset(data["X_train"], data["y_train"], mode)

    if policy_name == "none":
        train_dataset = base_train_dataset
    else:
        policy = get_augmentation_policy(policy_name)
        train_dataset = AugmentedSignalDataset(
            base_train_dataset,
            augmentation_policy=policy,
            augment_prob=0.8
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        SignalDataset(data["X_val"], data["y_val"], mode),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    ) if len(data["X_val"]) > 0 else None

    test_loader = DataLoader(
        SignalDataset(data["X_test"], data["y_test"], mode),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    # Create and train model
    num_classes = config.NUM_CLASSES[mode]
    model = create_model(model_name, num_classes, dropout=config.DROPOUT)
    model = model.to(device)

    if verbose:
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Training samples: {len(train_dataset)}")

    train_result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=mode,
        epochs=epochs,
        device=device,
        verbose=verbose,
    )

    # Evaluate - FIXED: correct argument order (device, mode)
    test_metrics = evaluate(model, test_loader, device, mode)

    result = {
        "model": model_name,
        "augmentation": policy_name,
        "mode": mode,
        "split": split,
        "seed": seed,
        "epochs": epochs,
        "test_metrics": test_metrics,
        "train_history": {
            "final_train_acc": train_result["history"]["train_acc"][-1],
            "final_train_loss": train_result["history"]["train_loss"][-1],
        },
        "timestamp": datetime.now().isoformat(),
    }

    if verbose:
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {test_metrics.get('macro_f1', 'N/A')}")

    return result


def run_sweep(
    models: list = None,
    policies: list = None,
    seeds: list = None,
    epochs: int = 100,
    verbose: bool = True,
):
    """Run full augmentation sweep."""
    if models is None:
        models = ["cnn"]
    if policies is None:
        policies = POLICIES
    if seeds is None:
        seeds = SEEDS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(models) * len(policies) * len(seeds)
    current = 0

    print_header("AUGMENTATION SWEEP")
    print(f"Total experiments: {total}")
    print(f"Models: {models}")
    print(f"Policies: {policies}")
    print(f"Seeds: {seeds}\n")

    for model_name in models:
        for policy_name in policies:
            for seed in seeds:
                current += 1
                print(f"[{current}/{total}] {model_name} + {policy_name} (seed={seed})")

                try:
                    result = run_single_experiment(
                        model_name=model_name,
                        policy_name=policy_name,
                        seed=seed,
                        epochs=epochs,
                        verbose=verbose,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    all_results.append({
                        "model": model_name,
                        "augmentation": policy_name,
                        "seed": seed,
                        "error": str(e),
                    })

    # Save results
    output_file = RESULTS_DIR / "augmentation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print_separator()
    print(f"Results saved to: {output_file}")

    # Print summary
    print_summary(all_results)

    return all_results


def print_summary(results: list):
    """Print summary of augmentation results."""
    from collections import defaultdict
    import numpy as np

    # Aggregate by policy
    policy_stats = defaultdict(list)
    for r in results:
        if "error" not in r:
            policy_stats[r["augmentation"]].append(r["test_metrics"]["accuracy"])

    print_separator()
    print("SUMMARY")
    print_separator()
    print(f"{'Policy':<12} {'Accuracy (%)':>12} {'Std':>10} {'N':>5}")
    print("-" * 42)

    summary = []
    for policy, accs in sorted(policy_stats.items()):
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        summary.append((policy, mean, std, len(accs)))

    for policy, mean, std, n in sorted(summary, key=lambda x: -x[1]):
        print(f"{policy:<12} {mean:>12.2f} {std:>10.2f} {n:>5}")

    if summary:
        best = max(summary, key=lambda x: x[1])
        print(f"\nBest: {best[0]} ({best[1]:.2f}% +- {best[2]:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run augmentation experiments")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "single"],
        default="quick",
        help="Experiment mode",
    )
    parser.add_argument("--policy", default="moderate", help="Policy for single mode")
    parser.add_argument("--seed", type=int, default=42, help="Seed for single mode")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.mode == "quick":
        run_sweep(
            policies=["none", "noise_only", "moderate"],
            seeds=[42],
            epochs=50,
            verbose=args.verbose,
        )
    elif args.mode == "full":
        run_sweep(
            seeds=SEEDS,
            epochs=args.epochs,
            verbose=args.verbose,
        )
    else:
        run_single_experiment(
            policy_name=args.policy,
            seed=args.seed,
            epochs=args.epochs,
            verbose=True,
        )


if __name__ == "__main__":
    main()
