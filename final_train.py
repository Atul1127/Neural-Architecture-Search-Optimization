"""
Train and evaluate the architecture discovered by RL-based NAS.
"""

import torch

from src.model import SearchCNN
from src.trainer import (
    get_cifar10_loaders,
    train_model,
    evaluate_model,
)


# Architecture selected by the 20-architecture,
# 3-epoch RL-NAS experiment.
BEST_ARCHITECTURE = {
    "filters": 32,
    "kernel_size": 5,
    "pooling": "avg",
    "activation": "gelu",
}

FINAL_EPOCHS = 10
LEARNING_RATE = 0.001


def count_parameters(model):
    """Count trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("FINAL RL-NAS ARCHITECTURE EVALUATION")
    print("=" * 60)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("\nSelected Architecture:")

    for key, value in BEST_ARCHITECTURE.items():
        print(f"  {key}: {value}")

    # Load CIFAR-10
    print("\nLoading CIFAR-10...")

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = get_cifar10_loaders(
        batch_size=128
    )

    # Build selected architecture
    model = SearchCNN(
        filters=BEST_ARCHITECTURE["filters"],
        kernel_size=BEST_ARCHITECTURE["kernel_size"],
        pooling=BEST_ARCHITECTURE["pooling"],
        activation=BEST_ARCHITECTURE["activation"],
    )

    parameter_count = count_parameters(model)

    print(
        f"\nTrainable Parameters: "
        f"{parameter_count:,}"
    )

    # Train from scratch
    print(
        f"\nTraining final model "
        f"for {FINAL_EPOCHS} epochs..."
    )

    model = train_model(
        model,
        train_loader,
        epochs=FINAL_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
    )

    # Validation evaluation
    print("\nEvaluating validation set...")

    validation_accuracy = evaluate_model(
        model,
        validation_loader,
        device=device,
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )

    # Final test evaluation
    print("\nEvaluating held-out test set...")

    test_accuracy = evaluate_model(
        model,
        test_loader,
        device=device,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"\nArchitecture:"
    )

    for key, value in BEST_ARCHITECTURE.items():
        print(f"  {key}: {value}")

    print(
        f"\nParameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
