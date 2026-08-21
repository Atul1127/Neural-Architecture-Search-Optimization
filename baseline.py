"""
Random Search baseline for Neural Architecture Search.

Uses the same search budget and efficiency-aware reward
as RL-NAS for a fair comparison.
"""

import random
import torch

from src.search_space import get_search_space
from src.model import SearchCNN
from src.trainer import (
    get_cifar10_loaders,
    train_model,
    evaluate_model,
)
from src.results import save_results


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEARCH_ITERATIONS = 20
CANDIDATE_EPOCHS = 1

PARAMETER_PENALTY = 0.10


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def sample_random_architecture(search_space):
    """Randomly sample one architecture."""

    return {
        key: random.choice(values)
        for key, values in search_space.items()
    }


def count_parameters(model):
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def calculate_reward(
    accuracy,
    parameter_count,
    max_parameters=300000,
):
    """Calculate the efficiency-aware reward."""

    accuracy_score = accuracy / 100.0

    parameter_ratio = min(
        parameter_count / max_parameters,
        1.0,
    )

    return (
        accuracy_score
        - PARAMETER_PENALTY * parameter_ratio
    )


# ---------------------------------------------------------
# Random Search
# ---------------------------------------------------------

def random_search(
    search_space,
    train_loader,
    validation_loader,
    device,
):
    """Run random architecture search."""

    best_architecture = None
    best_accuracy = 0.0
    best_parameters = 0
    best_reward = float("-inf")

    results = []

    print("\n" + "=" * 60)
    print("RANDOM SEARCH BASELINE")
    print("=" * 60)

    print(
        f"Search budget: "
        f"{SEARCH_ITERATIONS} architectures"
    )

    print(
        f"Parameter penalty: "
        f"{PARAMETER_PENALTY}"
    )

    for iteration in range(SEARCH_ITERATIONS):

        print("\n" + "=" * 60)

        print(
            f"Iteration "
            f"{iteration + 1}/{SEARCH_ITERATIONS}"
        )

        print("=" * 60)

        architecture = sample_random_architecture(
            search_space
        )

        print("\nGenerated Architecture:")

        for key, value in architecture.items():
            print(f"  {key}: {value}")

        model = SearchCNN(
            filters=architecture["filters"],
            kernel_size=architecture["kernel_size"],
            pooling=architecture["pooling"],
            activation=architecture["activation"],
        )

        parameter_count = count_parameters(model)

        model = train_model(
            model,
            train_loader,
            epochs=CANDIDATE_EPOCHS,
            device=device,
        )

        accuracy = evaluate_model(
            model,
            validation_loader,
            device=device,
        )

        reward = calculate_reward(
            accuracy,
            parameter_count,
        )

        print(
            f"\nValidation Accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            f"Trainable Parameters: "
            f"{parameter_count:,}"
        )

        print(
            f"Reward: "
            f"{reward:.4f}"
        )

        result = {
            "method": "Random Search",
            "iteration": iteration + 1,
            "architecture": architecture,
            "accuracy": accuracy,
            "parameters": parameter_count,
            "reward": reward,
        }

        results.append(result)

        if reward > best_reward:

            best_reward = reward
            best_accuracy = accuracy
            best_parameters = parameter_count
            best_architecture = architecture.copy()

            print("\nNew Best Architecture!")

    return (
        best_architecture,
        best_accuracy,
        best_parameters,
        best_reward,
        results,
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("NEURAL ARCHITECTURE SEARCH")
    print("RANDOM SEARCH BASELINE")
    print("=" * 60)

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("\nLoading CIFAR-10...")

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = get_cifar10_loaders(
        batch_size=128
    )

    print("CIFAR-10 loaded.")

    search_space = get_search_space()

    print("\nSearch Space:")

    for key, values in search_space.items():
        print(f"  {key}: {values}")

    (
        best_architecture,
        best_accuracy,
        best_parameters,
        best_reward,
        results,
    ) = random_search(
        search_space,
        train_loader,
        validation_loader,
        device,
    )

    # Save all results after the search completes.
    save_results(
        results,
        filename="random_search.csv",
    )

    print("\n" + "=" * 60)
    print("RANDOM SEARCH COMPLETE")
    print("=" * 60)

    print("\nBest Architecture:")

    for key, value in best_architecture.items():
        print(f"  {key}: {value}")

    print(
        f"\nBest Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Best Parameter Count: "
        f"{best_parameters:,}"
    )

    print(
        f"Best Efficiency Reward: "
        f"{best_reward:.4f}"
    )

    print(
        "\nResults saved to "
        "results/random_search.csv"
    )


if __name__ == "__main__":
    main()
