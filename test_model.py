import torch

from src.model import SearchCNN
from src.trainer import get_cifar10_loaders, train_model, evaluate_model


def main():
    train_loader, test_loader = get_cifar10_loaders()

    model = SearchCNN(
        filters=32,
        kernel_size=3,
        pooling="max",
        activation="relu",
    )

    print("Training model...")

    model = train_model(
        model,
        train_loader,
        epochs=1,
    )

    accuracy = evaluate_model(
        model,
        test_loader,
    )

    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
