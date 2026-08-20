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


def evaluate_architecture(
    architecture,
    train_loader,
    validation_loader,
    device,
    epochs=1,
):
    """
    Train and evaluate one candidate architecture.
    """

    model = SearchCNN(
        filters=architecture["filters"],
        kernel_size=architecture["kernel_size"],
        pooling=architecture["pooling"],
        activation=architecture["activation"],
    )

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

    return accuracy


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

    num_iterations = 5

    best_architecture = None
    best_accuracy = 0.0

    print("\nStarting search...\n")

    for iteration in range(num_iterations):

        print("=" * 60)
        print(
            f"Iteration "
            f"{iteration + 1}/{num_iterations}"
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

        # Train candidate and evaluate on validation set
        accuracy = evaluate_architecture(
            architecture,
            train_loader,
            validation_loader,
            device,
            epochs=1,
        )

        print(
            f"\nValidation Accuracy: "
            f"{accuracy:.2f}%"
        )

        # Normalize accuracy to [0, 1]
        reward = accuracy / 100.0

        # Update controller using REINFORCE
        loss = nas.update_controller(
            log_probability,
            reward,
        )

        print(
            f"Controller Loss: "
            f"{loss:.4f}"
        )

        # Track best architecture
        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_architecture = architecture

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


if __name__ == "__main__":
    main()
