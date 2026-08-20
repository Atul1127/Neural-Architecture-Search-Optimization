"""
Search space for CNN architectures.
"""

SEARCH_SPACE = {
    "filters": [16, 32, 64],
    "kernel_size": [3, 5],
    "pooling": ["max", "avg"],
    "activation": ["relu", "gelu"],
}


def get_search_space():
    return SEARCH_SPACE
