"""
Full experiment: Compare architectures on random vs fault-size splits.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

from config import config
from data import load_data, create_dataloaders
from models import get_model
from training import train_model, evaluate


def evaluate_with_metrics(model: nn.Module, loader, device: str) -> dict:
    """Evaluate model and return detailed metrics."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Get unique labels present in predictions and ground truth
    unique_labels = sorted(set(all_labels) | set(all_preds))
    label_names = [config["class_names"][i] for i in unique_labels]

    accuracy = (all_preds == all_labels).mean()
    report = classification_report(
        all_labels, all_preds,
        labels=unique_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
    }


def run_experiment(model_name: str, split_strategy: str, epochs: int = None) -> dict:
    """Run a single experiment."""
    if epochs is None:
        epochs = config["epochs"]

    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Split: {split_strategy}")
    print(f"{'='*60}")

    # Load data
    data = load_data(strategy=split_strategy)
    train_loader, val_loader, test_loader = create_dataloaders(data, config["batch_size"])

    # Create model
    model = get_model(model_name).to(config["device"])
    params = sum(p.numel() for p in model.parameters())

    # Train
    result = train_model(model, train_loader, val_loader, epochs=epochs, model_name=model_name)

    # Evaluate on test set
    test_metrics = evaluate_with_metrics(model, test_loader, config["device"])

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"\nPer-class results:")
    for cls_name in config["class_names"]:
        if cls_name in test_metrics["report"]:
            cls_report = test_metrics["report"][cls_name]
            print(f"  {cls_name}: P={cls_report['precision']:.3f} R={cls_report['recall']:.3f} F1={cls_report['f1-score']:.3f}")

    return {
        "model": model_name,
        "split_strategy": split_strategy,
        "parameters": params,
        "epochs_trained": len(result["history"]["train_loss"]),
        "best_val_acc": result["best_val_acc"],
        "test_accuracy": test_metrics["accuracy"],
        "classification_report": test_metrics["report"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "history": result["history"],
        "timestamp": datetime.now().isoformat(),
    }


def run_all_experiments(models: list = None, splits: list = None, epochs: int = None, seeds: list = None) -> list:
    """Run all experiment combinations."""
    if models is None:
        models = ["cnn1d", "lstm", "cnnlstm"]
    if splits is None:
        splits = ["random", "fault_size"]
    if seeds is None:
        seeds = [42]

    all_results = []

    for seed in seeds:
        config["seed"] = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        for split in splits:
            for model_name in models:
                result = run_experiment(model_name, split, epochs)
                result["seed"] = seed
                all_results.append(result)

    return all_results


def print_summary(results: list):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Model':<10} {'Split':<12} {'Params':<10} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['model']:<10} {r['split_strategy']:<12} {r['parameters']:<10,} "
              f"{r['best_val_acc']:<10.4f} {r['test_accuracy']:<10.4f}")

    print("=" * 70)

    # Calculate average drop
    random_results = [r for r in results if r["split_strategy"] == "random"]
    fault_results = [r for r in results if r["split_strategy"] == "fault_size"]

    if random_results and fault_results:
        avg_random = np.mean([r["test_accuracy"] for r in random_results])
        avg_fault = np.mean([r["test_accuracy"] for r in fault_results])
        print(f"\nAverage Test Accuracy:")
        print(f"  Random split:     {avg_random:.4f}")
        print(f"  Fault-size split: {avg_fault:.4f}")
        print(f"  Drop:             {avg_random - avg_fault:.4f} ({(avg_random - avg_fault) * 100:.1f}%)")


def save_results(results: list, filename: str = "results/experiment_results.json"):
    """Save results to JSON file."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # Run experiments
    results = run_all_experiments(
        models=["cnn1d", "lstm", "cnnlstm"],
        splits=["random", "fault_size"],
        epochs=50,
        seeds=[42],
    )

    # Print summary
    print_summary(results)

    # Save results
    save_results(results)
