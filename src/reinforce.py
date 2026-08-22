"""REINFORCE controller for neural architecture search."""

import torch
import torch.optim as optim

from src.controller import ArchitectureController
from src.search_space import get_search_space


class ReinforceNAS:
    """Sample CNN architectures and optimize the controller policy."""

    def __init__(self, train_loader, validation_loader, device=None, learning_rate=1e-3):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = ArchitectureController(hidden_size=64).to(self.device)
        self.optimizer = optim.Adam(self.controller.parameters(), lr=learning_rate)
        self.baseline = 0.0
        self.search_space = get_search_space()

    def sample_architecture(self):
        """Sample an architecture and return its summed log probability."""
        outputs = self.controller()
        architecture = {}
        log_probability = torch.tensor(0.0, device=self.device)

        for parameter, logits in outputs.items():
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            architecture[parameter] = self.search_space[parameter][action.item()]
            log_probability = log_probability + distribution.log_prob(action)

        return architecture, log_probability

    def update_controller(self, log_probability, reward):
        """Apply one REINFORCE update using a moving-average baseline."""
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.baseline = 0.9 * self.baseline + 0.1 * reward
        advantage = reward_tensor - self.baseline
        loss = -log_probability * advantage

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
