"""Definition of the CNN architecture search space."""

SEARCH_SPACE = {
    "filters": [16, 32, 64],
    "kernel_size": [3, 5],
    "pooling": ["max", "avg"],
    "activation": ["relu", "gelu"],
}


def get_search_space():
    """Return a copy so callers cannot mutate global configuration."""
    return {key: values.copy() for key, values in SEARCH_SPACE.items()}
