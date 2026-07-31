import json
import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return seed


def normalize_seeds(raw_seeds: Any) -> list[int]:
    """Normalize CLI/YAML seed values into a non-empty integer list.

    Some configuration parsers preserve command-line values such as
    ``seeds=[42]`` as strings. Accept scalar integers, integer sequences, JSON
    list strings, and comma-separated strings so experiment commands behave
    consistently.
    """

    if isinstance(raw_seeds, str):
        text = raw_seeds.strip()
        if not text:
            raise ValueError("seeds must not be empty")
        try:
            raw_seeds = json.loads(text)
        except json.JSONDecodeError:
            raw_seeds = [part.strip() for part in text.split(",") if part.strip()]

    if isinstance(raw_seeds, int):
        seeds = [raw_seeds]
    elif isinstance(raw_seeds, (list, tuple, set)):
        seeds = [int(seed) for seed in raw_seeds]
    else:
        raise TypeError(
            "seeds must be an integer, a sequence of integers, a JSON list, "
            "or a comma-separated string"
        )

    if not seeds:
        raise ValueError("seeds must contain at least one value")
    return seeds
