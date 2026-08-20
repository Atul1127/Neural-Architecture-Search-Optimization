"""
REINFORCE-based Neural Architecture Search.
"""

import torch
import torch.optim as optim

from src.controller import ArchitectureController


class ReinforceNAS:

    def __init__(
        self,
        train_loader,
        test_loader,
        device=None,
        learning_rate=0.001,
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.controller = ArchitectureController().to(self.device)

        self.optimizer = optim.Adam(
            self.controller.parameters(),
            lr=learning_rate,
        )

        self.baseline = 0.0

        self.search_space = {
            "filters": [16, 32, 64],
            "kernel_size": [3, 5],
            "pooling": ["max", "avg"],
            "activation": ["relu", "gelu"],
        }

    def sample_architecture(self):

        outputs = self.controller()

        architecture = {}
        log_probability = 0.0

        for parameter, logits in outputs.items():

            distribution = torch.distributions.Categorical(
                logits=logits
            )

            action = distribution.sample()

            log_probability += distribution.log_prob(action)

            architecture[parameter] = (
                self.search_space[parameter][action.item()]
            )

        return architecture, log_probability

    def update_controller(self, log_probability, reward):

        reward_tensor = torch.tensor(
            reward,
            dtype=torch.float32,
            device=self.device,
        )

        # Moving baseline reduces variance
        self.baseline = (
            0.9 * self.baseline +
            0.1 * reward
        )

        advantage = reward_tensor - self.baseline

        loss = -log_probability * advantage

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()
