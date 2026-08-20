"""
Random Search baseline for Neural Architecture Search.
"""

import torch

from src.nas import NeuralArchitectureSearch
from src.trainer import get_cifar10_loaders


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=128
    )

    nas = NeuralArchitectureSearch(
        train_loader,
        test_loader,
        device=device,
    )

    best_architecture, best_accuracy, results = (
        nas.random_search(
            num_architectures=5,
            epochs=1,
        )
    )

    print("\n" + "=" * 50)
    print("RANDOM SEARCH RESULTS")
    print("=" * 50)

    print("\nBest Architecture:")
    print(best_architecture)

    print(f"\nBest Accuracy: {best_accuracy:.2f}%")

    print("\nAll Results:")

    for i, result in enumerate(results, 1):
        print(
            f"{i}. "
            f"{result['architecture']} "
            f"-> {result['accuracy']:.2f}%"
        )


if __name__ == "__main__":
    main()
