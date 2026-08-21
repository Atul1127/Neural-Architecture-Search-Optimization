"""
Utilities for saving NAS experiment results.
"""

import csv
import os


def save_results(results, filename="results.csv"):
    """
    Save NAS experiment results to a CSV file.

    Each result contains:
        method
        architecture
        accuracy
        parameters
        reward
    """

    os.makedirs("results", exist_ok=True)

    filepath = os.path.join("results", filename)

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "method",
            "iteration",
            "filters",
            "kernel_size",
            "pooling",
            "activation",
            "accuracy",
            "parameters",
            "reward",
        ])

        for result in results:

            architecture = result["architecture"]

            writer.writerow([
                result["method"],
                result["iteration"],
                architecture["filters"],
                architecture["kernel_size"],
                architecture["pooling"],
                architecture["activation"],
                round(result["accuracy"], 4),
                result["parameters"],
                round(result["reward"], 6),
            ])

    print(f"Results saved to {filepath}")
