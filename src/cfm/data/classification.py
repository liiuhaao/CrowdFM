from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .base import TensorData


@dataclass
class ClassificationData(TensorData):
    num_worker: int
    num_task: int
    num_option: int
    dim: int
    triple: torch.Tensor
    task_y: torch.Tensor
    task_y_mask: torch.Tensor
    option_x: torch.Tensor | None = None

    def setup(self):
        if self.triple.dtype != torch.long or self.triple.ndim != 2 or self.triple.shape[0] != 3 or not self.triple.shape[1]:
            raise ValueError("triple must be a nonempty integer [3, num_responses] tensor")
        if self.task_y.shape != (self.num_task,) or self.task_y.dtype != torch.long:
            raise ValueError("classification truth must be an integer vector")
        if self.task_y_mask.shape != (self.num_task,) or self.task_y_mask.dtype != torch.bool:
            raise ValueError("truth mask must be an explicit boolean vector")
        workers, options, tasks = self.triple
        for ids, count in ((workers, self.num_worker), (options, self.num_option), (tasks, self.num_task)):
            if (ids < 0).any() or (ids >= count).any():
                raise ValueError("classification index out of range")
        truths = self.task_y[self.task_y_mask]
        if (truths < 0).any() or (truths >= self.num_option).any():
            raise ValueError("truth category out of range")
        self.worker_degree = torch.bincount(workers, minlength=self.num_worker)
        self.task_degree = torch.bincount(tasks, minlength=self.num_task)
        if (self.task_degree == 0).any():
            raise ValueError("each task must have at least one response")
        self.vote_counts = torch.zeros((self.num_task, self.num_option), device=self.triple.device)
        self.vote_counts.index_put_((tasks, options), torch.ones_like(tasks, dtype=self.vote_counts.dtype), accumulate=True)
        if self.option_x is None:
            self.option_x = torch.randn(self.num_option, self.dim, device=self.triple.device)
        return self


def get_classification_batch(scenes):
    if not scenes:
        raise ValueError("cannot batch an empty scene list")
    triples, task_slices = [], []
    nw = nt = no = 0
    for scene in scenes:
        offset = scene.triple.new_tensor([[nw], [no], [nt]])
        triples.append(scene.triple + offset)
        task_slices.append(slice(nt, nt + scene.num_task))
        nw += scene.num_worker
        nt += scene.num_task
        no += scene.num_option
    return SimpleNamespace(
        num_worker=nw, num_task=nt, triple=torch.cat(triples, dim=1),
        option_x=torch.cat([scene.option_x for scene in scenes]), task_slices=task_slices,
    )
