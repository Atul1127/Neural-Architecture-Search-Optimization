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
        validation_loader,
        device=None,
        learning_rate=0.001,
    ):

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # RNN controller
        self.controller = ArchitectureController(
            hidden_size=64
        ).to(self.device)

        # Optimizer for controller
        self.optimizer = optim.Adam(
            self.controller.parameters(),
            lr=learning_rate,
        )

        # Moving-average reward baseline
        self.baseline = 0.0

        # Architecture search space
        self.search_space = {
            "filters": [16, 32, 64],
            "kernel_size": [3, 5],
            "pooling": ["max", "avg"],
            "activation": ["relu", "gelu"],
        }

    def sample_architecture(self):
        """
        Generate an architecture using the RNN controller.

        Returns:
            architecture: sampled CNN architecture
            log_probability: log probability of sampled actions
        """

        outputs = self.controller()

        architecture = {}

        total_log_probability = torch.tensor(
            0.0,
            device=self.device,
        )

        for parameter, logits in outputs.items():

            # Convert controller output into probability distribution
            distribution = torch.distributions.Categorical(
                logits=logits
            )

            # Sample one architectural decision
            action = distribution.sample()

            # Store selected architecture value
            architecture[parameter] = (
                self.search_space[parameter][
                    action.item()
                ]
            )

            # Log probability required by REINFORCE
            total_log_probability = (
                total_log_probability
                + distribution.log_prob(action)
            )

        return (
            architecture,
            total_log_probability,
        )

    def update_controller(
        self,
        log_probability,
        reward,
    ):
        """
        Update the controller using REINFORCE.

        REINFORCE objective:

            Loss = -log(pi(a|s)) * advantage

        where:

            advantage = reward - baseline
        """

        reward_tensor = torch.tensor(
            reward,
            dtype=torch.float32,
            device=self.device,
        )

        # Update moving-average reward baseline
        self.baseline = (
            0.9 * self.baseline
            + 0.1 * reward
        )

        # Advantage tells us whether this architecture
        # performed better or worse than expected.
        advantage = (
            reward_tensor
            - self.baseline
        )

        # Policy-gradient loss
        loss = (
            -log_probability
            * advantage
        )

        # Update controller
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()
