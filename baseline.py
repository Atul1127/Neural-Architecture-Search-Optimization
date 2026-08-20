"""
Random Search baseline for Neural Architecture Search.
"""

import torch

from src.nas import NeuralArchitectureSearch
from src.trainer import get_cifar10_loaders


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("RANDOM SEARCH BASELINE")
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

    nas = NeuralArchitectureSearch(
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )

    best_architecture, best_accuracy, results = (
        nas.random_search(
            num_architectures=5,
            epochs=1,
        )
    )

    print("\n" + "=" * 60)
    print("RANDOM SEARCH RESULTS")
    print("=" * 60)

    print("\nBest Architecture:")

    for key, value in best_architecture.items():
        print(f"  {key}: {value}")

    print(
        f"\nBest Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print("\nAll Results:")

    for i, result in enumerate(results, 1):

        print(
            f"\n{i}. "
            f"{result['architecture']}"
        )

        print(
            f"   Accuracy: "
            f"{result['accuracy']:.2f}%"
        )


if __name__ == "__main__":
    main()
