"""
Random Search baseline for Neural Architecture Search.

The baseline uses the same search budget as RL-NAS so that
the comparison is fair.
"""

import torch

from src.search_space import SearchSpace
from src.model import SearchCNN
from src.trainer import (
    get_cifar10_loaders,
    train_model,
    evaluate_model,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEARCH_ITERATIONS = 20
CANDIDATE_EPOCHS = 1


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

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
    parameter_penalty=0.10,
):
    """
    Calculate the same efficiency-aware reward used by RL-NAS.

    Reward =
        normalized accuracy
        - parameter efficiency penalty
    """

    accuracy_score = accuracy / 100.0

    parameter_ratio = min(
        parameter_count / max_parameters,
        1.0,
    )

    reward = (
        accuracy_score
        - parameter_penalty * parameter_ratio
    )

    return reward


# ---------------------------------------------------------
# Random Search
# ---------------------------------------------------------

def random_search(
    search_space,
    train_loader,
    validation_loader,
    device,
):
    """
    Perform random architecture search.

    The search budget is intentionally identical to RL-NAS.
    """

    best_architecture = None
    best_accuracy = 0.0
    best_reward = float("-inf")
    best_parameters = 0

    results = []

    print("\n" + "=" * 60)
    print("RANDOM SEARCH BASELINE")
    print("=" * 60)

    print(
        f"Search budget: "
        f"{SEARCH_ITERATIONS} architectures"
    )

    for iteration in range(SEARCH_ITERATIONS):

        print("\n" + "-" * 60)

        print(
            f"Iteration "
            f"{iteration + 1}/{SEARCH_ITERATIONS}"
        )

        # Randomly sample an architecture
        architecture = search_space.sample()

        print("\nArchitecture:")

        for key, value in architecture.items():
            print(f"  {key}: {value}")

        # Create model
        model = SearchCNN(
            filters=architecture["filters"],
            kernel_size=architecture["kernel_size"],
            pooling=architecture["pooling"],
            activation=architecture["activation"],
        )

        # Count parameters BEFORE training
        parameter_count = count_parameters(model)

        # Train candidate
        model = train_model(
            model,
            train_loader,
            epochs=CANDIDATE_EPOCHS,
            device=device,
        )

        # Validation performance
        accuracy = evaluate_model(
            model,
            validation_loader,
            device=device,
        )

        # Same reward function as RL-NAS
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
            f"Efficiency Reward: "
            f"{reward:.4f}"
        )

        result = {
            "iteration": iteration + 1,
            "architecture": architecture,
            "accuracy": accuracy,
            "parameters": parameter_count,
            "reward": reward,
        }

        results.append(result)

        # Best architecture is selected using the SAME
        # efficiency-aware objective as RL-NAS.
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

    # Load CIFAR-10
    print("\nLoading CIFAR-10...")

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = get_cifar10_loaders(
        batch_size=128
    )

    print("CIFAR-10 loaded.")

    # Search space
    search_space = SearchSpace()

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

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

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


if __name__ == "__main__":
    main()
