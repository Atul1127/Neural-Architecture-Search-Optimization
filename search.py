"""Run the REINFORCE-based neural architecture search experiment."""

from src.config import BATCH_SIZE, CANDIDATE_EPOCHS, SEARCH_ITERATIONS
from src.model import SearchCNN
from src.reinforce import ReinforceNAS
from src.trainer import evaluate_model, get_cifar10_loaders, train_model
from src.utils import calculate_reward, count_parameters, get_device, seed_everything


def evaluate_architecture(architecture, train_loader, validation_loader, device):
    """Train one candidate architecture and return accuracy and parameter count."""
    model = SearchCNN(**architecture)
    parameter_count = count_parameters(model)
    model = train_model(model, train_loader, epochs=CANDIDATE_EPOCHS, device=device)
    accuracy = evaluate_model(model, validation_loader, device=device)
    return accuracy, parameter_count


def main():
    """Run the complete RL-NAS experiment."""
    seed_everything()
    device = get_device()

    print("=" * 60)
    print("NEURAL ARCHITECTURE SEARCH")
    print("=" * 60)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {__import__('torch').cuda.get_device_name(0)}")

    train_loader, validation_loader, _ = get_cifar10_loaders(batch_size=BATCH_SIZE)
    nas = ReinforceNAS(train_loader, validation_loader, device=device)

    best = {"architecture": None, "accuracy": 0.0, "parameters": 0, "reward": float("-inf")}
    print(f"\nSearch budget: {SEARCH_ITERATIONS} architectures")

    for iteration in range(SEARCH_ITERATIONS):
        print(f"\n{'=' * 60}\nIteration {iteration + 1}/{SEARCH_ITERATIONS}\n{'=' * 60}")
        architecture, log_probability = nas.sample_architecture()
        print("Generated Architecture:")
        for key, value in architecture.items():
            print(f"  {key}: {value}")

        accuracy, parameters = evaluate_architecture(
            architecture, train_loader, validation_loader, device
        )
        reward = calculate_reward(accuracy, parameters)
        loss = nas.update_controller(log_probability, reward)

        print(f"Validation Accuracy: {accuracy:.2f}%")
        print(f"Trainable Parameters: {parameters:,}")
        print(f"Reward: {reward:.4f}")
        print(f"Controller Loss: {loss:.4f}")

        if reward > best["reward"]:
            best = {
                "architecture": architecture.copy(),
                "accuracy": accuracy,
                "parameters": parameters,
                "reward": reward,
            }
            print("New Best Architecture!")

    print(f"\n{'=' * 60}\nSEARCH COMPLETE\n{'=' * 60}")
    print("Best Architecture:")
    for key, value in best["architecture"].items():
        print(f"  {key}: {value}")
    print(f"Best Validation Accuracy: {best['accuracy']:.2f}%")
    print(f"Best Parameter Count: {best['parameters']:,}")
    print(f"Best Efficiency Reward: {best['reward']:.4f}")


if __name__ == "__main__":
    main()
