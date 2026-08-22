"""Utilities for persisting NAS experiment results."""

import csv
from pathlib import Path

from src.config import RESULTS_DIR


def save_results(results, filename="results.csv"):
    """Save architecture-search results to a CSV file."""
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    fieldnames = [
        "method",
        "iteration",
        "filters",
        "kernel_size",
        "pooling",
        "activation",
        "accuracy",
        "parameters",
        "reward",
    ]

    with filepath.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            architecture = result["architecture"]
            writer.writerow({
                "method": result["method"],
                "iteration": result["iteration"],
                "filters": architecture["filters"],
                "kernel_size": architecture["kernel_size"],
                "pooling": architecture["pooling"],
                "activation": architecture["activation"],
                "accuracy": round(result["accuracy"], 4),
                "parameters": result["parameters"],
                "reward": round(result["reward"], 6),
            })

    print(f"Results saved to {filepath}")
