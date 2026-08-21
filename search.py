"""
Run Reinforcement Learning based Neural Architecture Search.
"""

import torch

from src.reinforce import ReinforceNAS
from src.trainer import (
    get_cifar10_loaders,
    train_model,
    evaluate_model,
)
from src.model import SearchCNN


# Search configuration
SEARCH_ITERATIONS = 20
CANDIDATE_EPOCHS = 1

# Weight for parameter efficiency.
# Higher = stronger preference for smaller models.
PARAMETER_PENALTY = 0.10


def count_parameters(model):
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def evaluate_architecture(
    architecture,
    train_loader,
    validation_loader,
    device,
    epochs=1,
):
    """
    Train and evaluate one candidate architecture.

    Returns:
        accuracy: validation accuracy in percentage
        parameter_count: number of trainable parameters
    """

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
        epochs=epochs,
        device=device,
    )

    accuracy = evaluate_model(
        model,
        validation_loader,
        device=device,
    )

    return accuracy, parameter_count


def calculate_reward(
    accuracy,
    parameter_count,
    max_parameters=300000,
):
    """
    Calculate accuracy-efficiency reward.

    Reward combines validation accuracy with a penalty
    for larger architectures.
    """

    accuracy_score = accuracy / 100.0

    parameter_ratio = min(
        parameter_count / max_parameters,
        1.0,
    )

    reward = (
        accuracy_score
        - PARAMETER_PENALTY * parameter_ratio
    )

    return reward


def main():
    """
    Run the complete RL-based NAS experiment.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("NEURAL ARCHITECTURE SEARCH")
    print("=" * 60)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
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

    nas = ReinforceNAS(
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )

    best_architecture = None
    best_accuracy = 0.0
    best_reward = float("-inf")
    best_parameters = 0

    print(
        f"\nSearch budget: "
        f"{SEARCH_ITERATIONS} architectures"
    )

    print(
        f"Parameter penalty: "
        f"{PARAMETER_PENALTY}"
    )

    print("\nStarting search...\n")

    for iteration in range(SEARCH_ITERATIONS):

        print("=" * 60)

        print(
            f"Iteration "
            f"{iteration + 1}/{SEARCH_ITERATIONS}"
        )

        print("=" * 60)

        # Controller generates an architecture
        (
            architecture,
            log_probability,
        ) = nas.sample_architecture()

        print("\nGenerated Architecture:")

        for key, value in architecture.items():
            print(f"  {key}: {value}")

        # Train candidate and evaluate
        (
            accuracy,
            parameter_count,
        ) = evaluate_architecture(
            architecture,
            train_loader,
            validation_loader,
            device,
            epochs=CANDIDATE_EPOCHS,
        )

        # Calculate efficiency-aware reward
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

        # Update controller using REINFORCE
        loss = nas.update_controller(
            log_probability,
            reward,
        )

        print(
            f"Controller Loss: "
            f"{loss:.4f}"
        )

        # Track architecture based on reward
        if reward > best_reward:

            best_reward = reward
            best_accuracy = accuracy
            best_parameters = parameter_count
            best_architecture = architecture.copy()

            print(
                "\nNew Best Architecture!"
            )

    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
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
