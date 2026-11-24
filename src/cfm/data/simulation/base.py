import numpy as np


class BaseCrowdSimulator:
    """Shared population ranges and sparse assignment graph only."""

    def __init__(self, **kwargs):
        allowed = {
            "num_worker_range", "num_task_range", "num_answer_each_task_range",
        }
        unknown = kwargs.keys() - allowed
        if unknown:
            raise ValueError(f"unknown simulator settings: {sorted(unknown)}; replay historical generators with archived source")
        self.num_worker_range = kwargs.get("num_worker_range", (20, 100))
        self.num_task_range = kwargs.get("num_task_range", (100, 300))
        self.num_answer_each_task_range = kwargs.get("num_answer_each_task_range", (2, 10))
        for name in ("num_worker_range", "num_task_range", "num_answer_each_task_range"):
            lower, upper = getattr(self, name)
            if lower < 1 or lower > upper:
                raise ValueError(f"invalid {name}")

    def _sample_graph(self, data):
        counts = np.clip(np.random.poisson(data.num_answer_each_task, data.num_task), 1, data.num_worker)
        permutations = np.random.rand(data.num_task, data.num_worker).argsort(axis=1)
        mask = counts[:, None] > np.arange(counts.max())[None, :]
        workers = permutations[:, :counts.max()][mask]
        tasks = np.repeat(np.arange(data.num_task), counts)
        return workers, tasks
