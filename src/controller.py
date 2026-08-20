"""
RNN controller for generating CNN architectures.
"""

import torch
import torch.nn as nn


class ArchitectureController(nn.Module):
    """
    RNN-based controller that generates CNN architecture decisions.
    """

    def __init__(self, hidden_size=64):
        super().__init__()

        self.hidden_size = hidden_size

        # LSTM controller
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Prediction heads for architecture decisions
        self.filters_head = nn.Linear(
            hidden_size,
            3,
        )

        self.kernel_head = nn.Linear(
            hidden_size,
            2,
        )

        self.pooling_head = nn.Linear(
            hidden_size,
            2,
        )

        self.activation_head = nn.Linear(
            hidden_size,
            2,
        )

        # Learnable starting representation
        self.start_token = nn.Parameter(
            torch.randn(
                1,
                1,
                hidden_size,
            )
        )

    def forward(self, batch_size=1):
        """
        Generate logits for each architecture decision.

        Returns:
            Dictionary containing logits for:
            filters, kernel size, pooling, activation.
        """

        x = self.start_token.expand(
            batch_size,
            1,
            self.hidden_size,
        )

        outputs, _ = self.rnn(x)

        hidden = outputs[:, -1, :]

        return {
            "filters": self.filters_head(hidden),
            "kernel_size": self.kernel_head(hidden),
            "pooling": self.pooling_head(hidden),
            "activation": self.activation_head(hidden),
        }
