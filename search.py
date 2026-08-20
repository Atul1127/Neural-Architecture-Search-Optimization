
"""
Run Reinforcement Learning based Neural Architecture Search.
"""

import torch

from src.reinforce import ReinforceNAS
from src.trainer import get_cifar10_loaders
from src.model import SearchCNN
from src.trainer import train_model, evaluate_model


def evaluate_architecture(
    architecture,
    train_loader,
    test_loader,
    device,
    epochs=1,
):
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
        test_loader,
        device=device,
    )

    return accuracy


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=128
    )

    nas = ReinforceNAS(
        train_loader,
        test_loader,
        device=device,
    )

    num_iterations = 5

    best_architecture = None
    best_accuracy = 0.0

    print("\nStarting Neural Architecture Search...\n")

    for iteration in range(num_iterations):

        print("=" * 50)
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print("=" * 50)

        architecture, log_probability = (
            nas.sample_architecture()
        )

        print("Generated architecture:")
        print(architecture)

        accuracy = evaluate_architecture(
            architecture,
            train_loader,
            test_loader,
            device,
            epochs=1,
        )

        print(f"Validation accuracy: {accuracy:.2f}%")

        # Accuracy acts as the reward
        reward = accuracy / 100.0

        loss = nas.update_controller(
            log_probability,
            reward,
        )

        print(f"Controller loss: {loss:.4f}")

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_architecture = architecture

            print("\nNew best architecture!")

    print("\n" + "=" * 50)
    print("SEARCH COMPLETE")
    print("=" * 50)

    print("\nBest Architecture:")
    print(best_architecture)

    print(f"\nBest Accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
