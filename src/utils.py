"""Shared NAS utilities."""

import random

import torch

from src.config import MAX_PARAMETERS, PARAMETER_PENALTY, RANDOM_SEED


def get_device():
    """Return the preferred computation device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model):
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_reward(accuracy, parameter_count):
    """Combine validation accuracy and parameter efficiency."""
    accuracy_score = accuracy / 100.0
    parameter_ratio = min(parameter_count / MAX_PARAMETERS, 1.0)
    return accuracy_score - PARAMETER_PENALTY * parameter_ratio


def seed_everything(seed=RANDOM_SEED):
    """Seed Python and PyTorch RNGs for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
