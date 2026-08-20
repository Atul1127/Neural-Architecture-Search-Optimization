"""
RNN controller for Neural Architecture Search.
"""

import torch
import torch.nn as nn


class ArchitectureController(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        return self.output(x)
