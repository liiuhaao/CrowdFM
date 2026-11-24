import random

import torch
import torch.nn.functional as F

from cfm.data.classification import get_classification_batch
from .encoder import BipartiteEncoder


class ClassificationModel(torch.nn.Module):
    def __init__(self, dim=32, layer=10, head=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.encoder = BipartiteEncoder(dim, layer, head, dropout)
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 2 * dim), torch.nn.LeakyReLU(), torch.nn.Linear(2 * dim, 1)
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for module in self.classification_head.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()

    def _logits(self, task_embeddings, option_embeddings):
        pairs = torch.cat((
            task_embeddings[:, None, :].expand(-1, option_embeddings.shape[0], -1),
            option_embeddings[None, :, :].expand(task_embeddings.shape[0], -1, -1),
        ), dim=-1)
        return self.classification_head(pairs).squeeze(-1)

    def forward(self, data):
        workers, options, tasks = data.triple
        _, zt = self.encoder(torch.stack((workers, tasks)), data.option_x[options], data.num_worker, data.num_task)
        return {"logits": self._logits(zt, data.option_x)}

    def loss(self, data, out=None):
        out = self(data) if out is None else out
        if not data.task_y_mask.any():
            raise ValueError("loss requires available truth")
        return F.cross_entropy(out["logits"][data.task_y_mask], data.task_y[data.task_y_mask])

    def batch_loss(self, scenes):
        batch = get_classification_batch(scenes)
        workers, options, tasks = batch.triple
        _, zt = self.encoder(torch.stack((workers, tasks)), batch.option_x[options], batch.num_worker, batch.num_task)
        losses = []
        for scene, selection in zip(scenes, batch.task_slices):
            if scene.task_y_mask.any():
                out = {"logits": self._logits(zt[selection], scene.option_x)}
                losses.append(self.loss(scene, out))
        if not losses:
            raise ValueError("batch has no available truth")
        return torch.stack(losses).mean()

    @torch.no_grad()
    def evaluate(self, data):
        out = self(data)
        mask = data.task_y_mask
        loss = self.loss(data, out)
        prediction = out["logits"].argmax(dim=1)
        majority = data.vote_counts.argmax(dim=1)
        tied = data.vote_counts.eq(data.vote_counts.max(dim=1, keepdim=True).values)
        for task in mask.nonzero(as_tuple=False).flatten().tolist():
            candidates = tied[task].nonzero(as_tuple=False).flatten().tolist()
            majority[task] = random.choice(candidates)
        return {
            "cross_entropy": loss.item(),
            "accuracy": (prediction[mask] == data.task_y[mask]).float().mean().item(),
            "majority_accuracy": (majority[mask] == data.task_y[mask]).float().mean().item(),
        }
