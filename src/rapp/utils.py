from typing import List
import numpy as np


def get_hidden_sizes(input_size: int, hidden_size: int, n_layers: int) -> List[int]:
    return list(np.linspace(input_size, hidden_size, n_layers + 1).astype(int))
