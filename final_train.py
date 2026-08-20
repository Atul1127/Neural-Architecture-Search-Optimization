"""
Train and evaluate the best architecture discovered by NAS.
"""

import torch

from src.model import SearchCNN
from src.trainer import (
    get_cifar10_loaders,
    train_model,
    evaluate_model,
)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("FINAL ARCHITECTURE EVALUATION")
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

    # Best architecture discovered during NAS
    best_architecture = {
        "filters": 64,
        "kernel_size": 5,
        "pooling": "max",
        "activation": "gelu",
    }

    print("\nSelected Architecture:")

    for key, value in best_architecture.items():
        print(f"  {key}: {value}")

    model = SearchCNN(
        filters=best_architecture["filters"],
        kernel_size=best_architecture["kernel_size"],
        pooling=best_architecture["pooling"],
        activation=best_architecture["activation"],
    )

    print("\nTraining final model...")

    model = train_model(
        model,
        train_loader,
        epochs=10,
        lr=0.001,
        device=device,
    )

    print("\nEvaluating on test set...")

    test_accuracy = evaluate_model(
        model,
        test_loader,
        device=device,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"\nTest Accuracy: "
        f"{test_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()
