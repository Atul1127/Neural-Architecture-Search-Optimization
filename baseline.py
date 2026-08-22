"""Run a Random Search baseline with the same NAS budget."""

import random

from src.config import BATCH_SIZE, CANDIDATE_EPOCHS, SEARCH_ITERATIONS
from src.model import SearchCNN
from src.results import save_results
from src.search_space import get_search_space
from src.trainer import evaluate_model, get_cifar10_loaders, train_model
from src.utils import calculate_reward, count_parameters, get_device, seed_everything


def sample_random_architecture(search_space):
    """Sample one architecture uniformly from the search space."""
    return {key: random.choice(values) for key, values in search_space.items()}


def random_search(search_space, train_loader, validation_loader, device):
    """Evaluate random candidate architectures and track the best reward."""
    best = {"architecture": None, "accuracy": 0.0, "parameters": 0, "reward": float("-inf")}
    results = []

    for iteration in range(SEARCH_ITERATIONS):
        architecture = sample_random_architecture(search_space)
        model = SearchCNN(**architecture)
        parameters = count_parameters(model)
        model = train_model(model, train_loader, epochs=CANDIDATE_EPOCHS, device=device)
        accuracy = evaluate_model(model, validation_loader, device=device)
        reward = calculate_reward(accuracy, parameters)

        results.append({
            "method": "Random Search",
            "iteration": iteration + 1,
            "architecture": architecture,
            "accuracy": accuracy,
            "parameters": parameters,
            "reward": reward,
        })

        print(
            f"Iteration {iteration + 1}/{SEARCH_ITERATIONS} | "
            f"accuracy={accuracy:.2f}% | parameters={parameters:,} | reward={reward:.4f}"
        )

        if reward > best["reward"]:
            best = {
                "architecture": architecture.copy(),
                "accuracy": accuracy,
                "parameters": parameters,
                "reward": reward,
            }
            print("  -> New best architecture")

    return best, results


def main():
    """Run and persist the Random Search baseline."""
    seed_everything()
    device = get_device()
    train_loader, validation_loader, _ = get_cifar10_loaders(batch_size=BATCH_SIZE)
    search_space = get_search_space()

    print("=" * 60)
    print("RANDOM SEARCH BASELINE")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Search budget: {SEARCH_ITERATIONS} architectures")

    best, results = random_search(search_space, train_loader, validation_loader, device)
    save_results(results, filename="random_search.csv")

    print("\nBest Architecture:")
    for key, value in best["architecture"].items():
        print(f"  {key}: {value}")
    print(f"Best Validation Accuracy: {best['accuracy']:.2f}%")
    print(f"Best Parameter Count: {best['parameters']:,}")
    print(f"Best Efficiency Reward: {best['reward']:.4f}")


if __name__ == "__main__":
    main()
