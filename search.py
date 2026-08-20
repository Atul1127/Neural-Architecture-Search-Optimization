"""
Random Search baseline for Neural Architecture Search.
"""

import random

from src.model import SearchCNN
from src.trainer import train_model, evaluate_model


class NeuralArchitectureSearch:

    def __init__(
        self,
        train_loader,
        validation_loader,
        device=None,
    ):

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.device = device

        self.search_space = {
            "filters": [16, 32, 64],
            "kernel_size": [3, 5],
            "pooling": ["max", "avg"],
            "activation": ["relu", "gelu"],
        }

    def sample_architecture(self):

        architecture = {
            "filters": random.choice(
                self.search_space["filters"]
            ),

            "kernel_size": random.choice(
                self.search_space["kernel_size"]
            ),

            "pooling": random.choice(
                self.search_space["pooling"]
            ),

            "activation": random.choice(
                self.search_space["activation"]
            ),
        }

        return architecture

    def evaluate_architecture(
        self,
        architecture,
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
            self.train_loader,
            epochs=epochs,
            device=self.device,
        )

        accuracy = evaluate_model(
            model,
            self.validation_loader,
            device=self.device,
        )

        return accuracy

    def random_search(
        self,
        num_architectures=5,
        epochs=1,
    ):

        best_architecture = None
        best_accuracy = 0.0

        results = []

        for i in range(num_architectures):

            print(
                f"\nArchitecture "
                f"{i + 1}/{num_architectures}"
            )

            architecture = self.sample_architecture()

            print(
                "Architecture:",
                architecture,
            )

            accuracy = self.evaluate_architecture(
                architecture,
                epochs=epochs,
            )

            print(
                f"Validation Accuracy: "
                f"{accuracy:.2f}%"
            )

            results.append({
                "architecture": architecture,
                "accuracy": accuracy,
                "method": "Random Search",
            })

            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_architecture = architecture

        return (
            best_architecture,
            best_accuracy,
            results,
        )
