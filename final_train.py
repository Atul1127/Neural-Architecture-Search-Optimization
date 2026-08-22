"""Retrain and evaluate the architecture selected by RL-NAS."""

from src.config import BATCH_SIZE, FINAL_EPOCHS, LEARNING_RATE
from src.model import SearchCNN
from src.trainer import evaluate_model, get_cifar10_loaders, train_model
from src.utils import count_parameters, get_device, seed_everything


BEST_ARCHITECTURE = {
    "filters": 32,
    "kernel_size": 5,
    "pooling": "avg",
    "activation": "gelu",
}


def main():
    """Retrain the selected architecture and evaluate on validation/test sets."""
    seed_everything()
    device = get_device()
    train_loader, validation_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)

    model = SearchCNN(**BEST_ARCHITECTURE)
    parameters = count_parameters(model)

    print("=" * 60)
    print("FINAL RL-NAS ARCHITECTURE EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Architecture: {BEST_ARCHITECTURE}")
    print(f"Trainable parameters: {parameters:,}")
    print(f"Training epochs: {FINAL_EPOCHS}")

    model = train_model(
        model,
        train_loader,
        epochs=FINAL_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
    )

    validation_accuracy = evaluate_model(model, validation_loader, device=device)
    test_accuracy = evaluate_model(model, test_loader, device=device)

    print("\nFinal Results")
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
