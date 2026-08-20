"""
Utilities for saving NAS experiment results.
"""

import csv
import os


def save_results(results, filename="results.csv"):

    os.makedirs("results", exist_ok=True)

    filepath = os.path.join("results", filename)

    with open(filepath, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "method",
            "filters",
            "kernel_size",
            "pooling",
            "activation",
            "accuracy",
        ])

        for result in results:

            architecture = result["architecture"]

            writer.writerow([
                result["method"],
                architecture["filters"],
                architecture["kernel_size"],
                architecture["pooling"],
                architecture["activation"],
                result["accuracy"],
            ])

    print(f"Results saved to {filepath}")
